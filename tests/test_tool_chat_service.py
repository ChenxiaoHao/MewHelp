import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from app.schemas.chat import ChatMessage
from app.services import tool_chat_service as svc
from app.tools.executor import ToolOutcome


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    from app.core.config import Settings

    return Settings(_env_file=None)


class FakeToolModel:
    """两轮各给一段流脚本；第二轮记录收到的完整 messages（验证回灌）。"""

    def __init__(self, round1, round2):
        self.scripts = [round1, round2]
        self.calls = 0
        self.round2_messages = None
        self.bound_tools = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    async def astream(self, messages, **kwargs):
        idx = self.calls
        self.calls += 1
        if idx == 1:
            self.round2_messages = list(messages)
        for c in self.scripts[idx]:
            yield c


class FakePersister:
    def __init__(self):
        self.calls = []

    async def on_tool_calls(self, conversation_id, content, tool_calls):
        self.calls.append(("tool_calls", conversation_id, content, tool_calls))

    async def on_tool_result(self, conversation_id, outcome):
        self.calls.append(("tool_result", conversation_id, outcome))

    async def on_final_answer(self, conversation_id, content):
        self.calls.append(("final", conversation_id, content))


def user_msg(text="你好"):
    return [ChatMessage(role="user", content=text)]


async def collect(agen):
    return [e async for e in agen]


def tc_chunk(args_fragment, first=False):
    """构造 tool_call 分片 chunk（first=True 带 name/id，模拟真实上游拆包）。"""
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "query_logistics" if first else None,
                "args": args_fragment,
                "id": "call_1" if first else None,
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )


def faq_chunk(args_fragment, first=False):
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "query_faq" if first else None, "args": args_fragment,
                           "id": "call_f" if first else None, "index": 0, "type": "tool_call_chunk"}],
    )


def faq_outcome(refused=False, hits=None, note=""):
    return ToolOutcome(
        "query_faq", "call_f", True,
        {"keyword": "退货款几天到账", "hits": hits or [], "refused": refused, "note": note},
        "未命中" if not hits else f"命中 {len(hits)} 条",
    )


HIT1 = [{"n": 1, "id": 5, "question": "q", "answer": "a", "category": "c", "section_path": "s"}]


async def _ret(v):
    """把同步值包成 awaitable（给 monkeypatch 的 lambda 用）。"""
    return v


async def test_pure_chat_passthrough(settings):
    """无工具决策：token 实时透传，第二轮根本不发生，行为与 ch01 等价。"""
    model = FakeToolModel(
        round1=[AIMessageChunk(content="你好"), AIMessageChunk(content="呀")],
        round2=[],
    )
    p = FakePersister()
    events = await collect(
        svc.stream_chat_with_tools(
            user_msg(), settings, model, conversation_id=1, persister=p
        )
    )
    assert events == [("token", "你好"), ("token", "呀")]
    assert model.calls == 1
    assert model.round2_messages is None
    assert len(model.bound_tools) == 4  # ch08 内置四件套(logistics 归 MCP)
    assert ("final", 1, "你好呀") in p.calls
    assert [c[0] for c in p.calls] == ["final"]


async def test_tool_flow_events_and_round2_feedback(settings, monkeypatch):
    model = FakeToolModel(
        round1=[tc_chunk('{"order_id": ', first=True), tc_chunk('"1001"}')],
        round2=[AIMessageChunk(content="您的包裹"), AIMessageChunk(content="在杭州")],
    )

    async def fake_execute(name, args, tcid, ctx):
        assert name == "query_logistics"
        assert args == {"order_id": "1001"}  # 分片 args 已聚合解析
        assert tcid == "call_1"
        assert ctx.conversation_id == 3
        assert ctx.timeout_seconds == settings.tool_timeout_seconds
        assert ctx.max_retries == settings.tool_max_retries
        return ToolOutcome(
            name=name, tool_call_id=tcid, ok=True,
            result={"current_status": "运输中"}, summary="运输中",
        )

    monkeypatch.setattr(svc, "execute_tool", fake_execute)
    p = FakePersister()
    events = await collect(
        svc.stream_chat_with_tools(
            user_msg("订单1001物流到哪了"), settings, model,
            conversation_id=3, persister=p,
        )
    )
    kinds = [k for k, _ in events]
    assert kinds == ["tool_call", "tool_result", "token", "token"]
    assert events[0][1] == {"id": "call_1", "name": "query_logistics", "args": {"order_id": "1001"}}
    assert events[1][1] == {"id": "call_1", "name": "query_logistics", "ok": True, "summary": "运输中"}

    # 第二轮回灌：裸 model 收到 [...历史, AIMessage(含tool_calls), ToolMessage(结果JSON)]
    assert model.calls == 2
    r2 = model.round2_messages
    tm = [m for m in r2 if isinstance(m, ToolMessage)]
    assert len(tm) == 1
    assert tm[0].tool_call_id == "call_1"
    assert "运输中" in tm[0].content
    ai = [m for m in r2 if getattr(m, "tool_calls", None)]
    assert ai and ai[0].tool_calls[0]["name"] == "query_logistics"

    # 落库三挂点按序各一次
    assert [c[0] for c in p.calls] == ["tool_calls", "tool_result", "final"]
    assert p.calls[0][2] == ""  # 第一轮无文本
    assert p.calls[0][3][0]["name"] == "query_logistics"
    assert p.calls[2][2] == "您的包裹在杭州"


async def test_tool_failure_still_converges(settings, monkeypatch):
    """工具失败：ok=False 帧照发，错误 JSON 照样回灌，第二轮礼貌收敛（spec §9）。"""
    model = FakeToolModel(
        round1=[tc_chunk('{"order_id": "1001"}', first=True)],
        round2=[AIMessageChunk(content="抱歉，暂时查询不到")],
    )

    async def fake_execute(name, args, tcid, ctx):
        return ToolOutcome(
            name=name, tool_call_id=tcid, ok=False,
            result={"error": "工具执行失败: 执行超时(>5.0s)"}, summary="失败: 执行超时",
        )

    monkeypatch.setattr(svc, "execute_tool", fake_execute)
    events = await collect(
        svc.stream_chat_with_tools(user_msg("查订单"), settings, model, conversation_id=5)
    )
    kinds = [k for k, _ in events]
    assert kinds == ["tool_call", "tool_result", "token"]
    assert events[1][1]["ok"] is False
    r2 = model.round2_messages
    tm = [m for m in r2 if isinstance(m, ToolMessage)][0]
    assert "error" in tm.content


async def test_persister_failure_never_breaks_stream(settings):
    model = FakeToolModel(round1=[AIMessageChunk(content="好")], round2=[])

    class BrokenPersister:
        async def on_tool_calls(self, *a):
            raise RuntimeError("db down")

        async def on_tool_result(self, *a):
            raise RuntimeError("db down")

        async def on_final_answer(self, *a):
            raise RuntimeError("db down")

    events = await collect(
        svc.stream_chat_with_tools(
            user_msg(), settings, model, conversation_id=1, persister=BrokenPersister()
        )
    )
    assert events == [("token", "好")]  # 落库炸了流也要完整


async def test_no_persister_is_noop(settings):
    model = FakeToolModel(round1=[AIMessageChunk(content="喵")], round2=[])
    events = await collect(
        svc.stream_chat_with_tools(user_msg(), settings, model)
    )
    assert events == [("token", "喵")]


async def test_gate1_refused_exits_with_fixed_refusal_no_round2(settings, monkeypatch):
    model = FakeToolModel(
        round1=[faq_chunk('{"keyword": ', first=True), faq_chunk('"退货款几天到账"}')],
        round2=[AIMessageChunk(content="不该走到第二轮")],
    )

    async def fake_execute(name, args, tcid, ctx):
        return faq_outcome(refused=True, note="证据置信度不足")

    monkeypatch.setattr(svc, "execute_tool", fake_execute)
    p = FakePersister()
    events = await collect(svc.stream_chat_with_tools(
        user_msg("退货款几天到账"), settings, model, conversation_id=9, persister=p))
    assert [k for k, _ in events] == ["tool_call", "tool_result", "token"]
    assert events[-1][1] == svc.REFUSAL_ANSWER
    assert model.calls == 1
    assert [c[0] for c in p.calls] == ["tool_calls", "tool_result", "final"]
    assert p.calls[-1][2] == svc.REFUSAL_ANSWER  # 固定文案落库(与 yield 同源,§5.2)


async def test_gate2_insufficient_refuses_and_pools(settings, monkeypatch):
    from app.services.self_check import EvidenceCheck

    model = FakeToolModel(
        round1=[faq_chunk('{"keyword": ', first=True), faq_chunk('"退货款几天到账"}')], round2=[])
    seen = {}

    async def fake_execute(name, args, tcid, ctx):
        return faq_outcome(hits=HIT1)

    async def fake_check(question, hits, st, *, model=None):
        seen["q"] = question
        return EvidenceCheck(sufficient=False, reason="证据只沾边")

    async def fake_pool(cid, q, source, reason):
        seen.update(cid=cid, source=source, reason=reason)

    monkeypatch.setattr(svc, "execute_tool", fake_execute)
    monkeypatch.setattr(svc, "evaluate_evidence", fake_check)
    monkeypatch.setattr(svc, "pool_low_confidence", fake_pool)
    events = await collect(svc.stream_chat_with_tools(
        user_msg("退货款几天到账"), settings, model, conversation_id=11, persister=None))
    assert events[-1] == ("token", svc.REFUSAL_ANSWER)
    assert model.calls == 1  # 不进第二轮
    assert seen == {"q": "退货款几天到账", "cid": 11, "source": "self_check", "reason": "证据只沾边"}


async def test_gate2_verdict_none_passes_through(settings, monkeypatch):
    model = FakeToolModel(
        round1=[faq_chunk('{"keyword": ', first=True), faq_chunk('"退货款几天到账"}')],
        round2=[AIMessageChunk(content="最终回答")],
    )
    monkeypatch.setattr(svc, "execute_tool",
                        lambda *a: _ret(faq_outcome(hits=HIT1)))

    async def fake_check(question, hits, st, *, model=None):
        return None  # 自评失败 = 放行(§0-2)
    monkeypatch.setattr(svc, "evaluate_evidence", fake_check)
    events = await collect(svc.stream_chat_with_tools(
        user_msg("退货款几天到账"), settings, model, conversation_id=None, persister=None))
    assert model.calls == 2 and events[-1] == ("token", "最终回答")


async def test_gate2_disabled_skips_check_entirely(settings, monkeypatch):
    st = settings.model_copy(update={"self_check_enabled": False})
    model = FakeToolModel(
        round1=[faq_chunk('{"keyword": ', first=True), faq_chunk('"退货款几天到账"}')],
        round2=[AIMessageChunk(content="最终回答")],
    )
    monkeypatch.setattr(svc, "execute_tool", lambda *a: _ret(faq_outcome(hits=HIT1)))

    async def never_called(*a, **k):
        raise AssertionError("开关关闭时不得触发自评")

    monkeypatch.setattr(svc, "evaluate_evidence", never_called)
    events = await collect(svc.stream_chat_with_tools(
        user_msg("退货款几天到账"), st, model, conversation_id=None, persister=None))
    assert events[-1] == ("token", "最终回答")
