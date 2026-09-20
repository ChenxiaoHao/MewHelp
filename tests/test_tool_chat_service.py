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
    assert len(model.bound_tools) == 5  # 第一轮确实 bind 了 5 个工具
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
