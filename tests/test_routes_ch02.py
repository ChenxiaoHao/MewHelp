from langchain_core.messages import AIMessageChunk


class FakeConv:
    id = 7


class FakeCrud:
    """routes 与 persistence 共用的 crud 替身：记录所有写入。"""

    def __init__(self):
        self.messages = []
        self.conv_calls = []

    async def create_or_get_conversation(self, session, conversation_id, user_id):
        self.conv_calls.append((conversation_id, user_id))
        return FakeConv()

    async def add_message(
        self, session, conversation_id, role, content=None, tool_calls=None, tool_call_id=None
    ):
        self.messages.append((conversation_id, role, content, tool_calls, tool_call_id))


class ScriptedToolModel:
    """第一轮吐 tool_call 分片，第二轮（裸模型）吐文本。"""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "query_faq",
                        "args": '{"keyword": "退货政策"}',
                        "id": "c1",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            )
        else:
            yield AIMessageChunk(content="7天无理由退换")


def _override(app, dep, value):
    app.dependency_overrides[dep] = lambda: value


# ---- P5 回归基线:ch05 起 /api/chat/stream 换 Graph 接线,以下测试从端点面
# 重定向到服务层 stream_chat_with_tools(ch04 编排语义不变);端点帧面
# (conversation/降级/error/落库 bootstrap)回归归 tests/test_chat_stream_ch05.py。----


def _msgs(text):
    from app.schemas.chat import ChatMessage

    return [ChatMessage(role="user", content=text)]


async def _collect(agen):
    return [ev async for ev in agen]


async def test_no_db_degrades_to_ch01_frames(fake_settings):
    """persister=None(无库降级输入)→ 纯 token 事件流与 ch01 等价,不落库不炸。"""
    from app.services.tool_chat_service import stream_chat_with_tools
    from tests.conftest import FakeChatModel

    events = await _collect(
        stream_chat_with_tools(_msgs("你好"), fake_settings, FakeChatModel())
    )
    assert {k for k, _ in events} == {"token"}
    assert "".join(d for k, d in events if k == "token") == "你好呀喵"


async def test_full_frames_with_db(fake_settings, monkeypatch):
    """DBChatPersister 挂点经服务层直连:on_final_answer 落 assistant 行。"""
    from app.services import persistence as pers_mod
    from app.services.persistence import DBChatPersister
    from app.services.tool_chat_service import stream_chat_with_tools
    from tests.conftest import FakeChatModel

    fake_crud = FakeCrud()
    monkeypatch.setattr(pers_mod, "crud", fake_crud)
    events = await _collect(
        stream_chat_with_tools(_msgs("你好"), fake_settings, FakeChatModel(),
                               conversation_id=7, persister=DBChatPersister(object(), 7))
    )
    assert "".join(d for k, d in events if k == "token") == "你好呀喵"
    assert fake_crud.messages == [(7, "assistant", "你好呀喵", None, None)]


async def test_tool_frames_order_and_shape(fake_settings, monkeypatch):
    """事件序 tool_call→tool_result→token 与帧 payload 形状（spec §6）。"""
    from app.services import tool_chat_service as svc
    from app.services.tool_chat_service import stream_chat_with_tools
    from app.tools.executor import ToolOutcome

    async def fake_execute(spec, args, tcid, ctx):
        assert ctx.conversation_id == 7  # 会话 id 已注入工具上下文
        return ToolOutcome(spec.name, tcid, True, {"keyword": "退货政策", "hits": [{"question": "退货政策是什么？"}]}, "命中 1 条")

    monkeypatch.setattr(svc, "execute_tool", fake_execute)

    async def fake_check(question, hits, settings, *, model=None):
        return None  # 闸2 = 不判即放行(降级语义);基线帧断言不触真实自评
    monkeypatch.setattr(svc, "evaluate_evidence", fake_check)

    events = await _collect(
        stream_chat_with_tools(_msgs("退货政策是什么"), fake_settings,
                               ScriptedToolModel(), conversation_id=7)
    )
    assert [k for k, _ in events] == ["tool_call", "tool_result", "token"]
    assert events[0][1] == {"id": "c1", "name": "query_faq", "args": {"keyword": "退货政策"}}
    assert events[1][1] == {"id": "c1", "name": "query_faq", "ok": True, "summary": "命中 1 条"}


async def test_tool_result_frame_carries_citations(fake_settings, monkeypatch):
    """ch04: query_faq 命中帧带 citations 键;前端弹窗数据链路。"""
    from app.services import tool_chat_service as svc
    from app.services.tool_chat_service import stream_chat_with_tools
    from app.tools.executor import ToolOutcome

    cites = [{"n": 1, "chunk_id": 7, "section_path": "手册 > 节1", "question": "q", "answer": "a"}]

    async def fake_execute(spec, args, tcid, ctx):
        return ToolOutcome(spec.name, tcid, True, {"keyword": "k", "hits": [{"n": 1}], "refused": False, "note": ""},
                           "命中 1 条", citations=cites)

    monkeypatch.setattr(svc, "execute_tool", fake_execute)

    async def fake_check(question, hits, settings, *, model=None):
        return None  # 闸2 = 不判即放行(降级语义)
    monkeypatch.setattr(svc, "evaluate_evidence", fake_check)

    events = await _collect(
        stream_chat_with_tools(_msgs("退货政策"), fake_settings, ScriptedToolModel())
    )
    tr = [d for k, d in events if k == "tool_result"][0]
    assert tr["citations"] == cites


async def test_conversation_id_zero_rejected_422(client):
    from app.main import app  # noqa: F401 —— 确保 app 已构建

    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}], "conversation_id": 0},
    )
    assert r.status_code == 422


async def test_ch01_style_request_without_conversation_id_still_200(client, monkeypatch):
    """不带 conversation_id 的 ch01 请求体依然兼容（spec §6）：bootstrap 自动建
    一次性会话 + conversation/token/done 帧齐。ch05 起闲聊走 fast path 固定话术,
    token 帧数不再是断言面(内容回归归 ch05 测)。"""
    from app.api import routes as routes_mod
    from app.main import app
    from app.services import persistence as pers_mod

    fake_crud = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake_crud)
    monkeypatch.setattr(pers_mod, "crud", fake_crud)
    _override(app, routes_mod.dep_db_session, object())

    r = await client.post(
        "/api/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 200
    assert "event: conversation" in r.text
    assert "event: token" in r.text and "[DONE]" in r.text
    assert fake_crud.conv_calls == [(None, "demo_user")]  # 自动建一次性会话
