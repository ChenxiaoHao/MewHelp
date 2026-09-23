import json

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


async def test_no_db_degrades_to_ch01_frames(client):
    """引擎未初始化 → dep_db_session yield None：无 conversation 帧、无落库，token/done 与 ch01 完全一致。"""
    from app.api.routes import dep_chat_model
    from app.main import app
    from tests.conftest import FakeChatModel

    _override(app, dep_chat_model, FakeChatModel())
    r = await client.post(
        "/api/chat/stream", json={"messages": [{"role": "user", "content": "你好"}]}
    )
    assert r.status_code == 200
    assert "event: conversation" not in r.text
    assert "event: tool_call" not in r.text
    data = [
        line[len("data:"):].strip()
        for line in r.text.splitlines()
        if line.startswith("data:") and line[len("data:"):].strip().startswith('"')
    ]
    assert "".join(json.loads(d) for d in data) == "你好呀喵"
    assert "[DONE]" in r.text


async def test_full_frames_with_db(client, monkeypatch):
    """帧序 conversation→token→done；on_turn_start 落 user 行；on_final_answer 落 assistant 行。"""
    from app.api import routes as routes_mod
    from app.main import app
    from app.services import persistence as pers_mod
    from tests.conftest import FakeChatModel

    fake_crud = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake_crud)
    monkeypatch.setattr(pers_mod, "crud", fake_crud)
    _override(app, routes_mod.dep_db_session, object())
    _override(app, routes_mod.dep_chat_model, FakeChatModel())

    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "你好"}], "conversation_id": 7},
    )
    body = r.text
    assert body.index("event: conversation") < body.index("event: token")
    conv_lines = [
        line for line in body.splitlines()
        if line.startswith("data:") and "conversation_id" in line
    ]
    assert json.loads(conv_lines[0][len("data:"):].strip()) == {"conversation_id": 7}
    assert fake_crud.conv_calls == [(7, "demo_user")]
    assert (7, "user", "你好", None, None) in fake_crud.messages
    assert any(m[1] == "assistant" and m[2] == "你好呀喵" for m in fake_crud.messages)
    assert "[DONE]" in body


async def test_tool_frames_order_and_shape(client, monkeypatch):
    """帧序 conversation→tool_call→tool_result→token→done（spec §6）。"""
    from app.api import routes as routes_mod
    from app.main import app
    from app.services import tool_chat_service as svc
    from app.tools.executor import ToolOutcome

    monkeypatch.setattr(routes_mod, "crud", FakeCrud())
    _override(app, routes_mod.dep_db_session, object())
    _override(app, routes_mod.dep_chat_model, ScriptedToolModel())

    async def fake_execute(name, args, tcid, ctx):
        assert ctx.conversation_id == 7  # conversation 帧的 id 已注入工具上下文
        return ToolOutcome(name, tcid, True, {"keyword": "退货政策", "hits": [{"question": "退货政策是什么？"}]}, "命中 1 条")

    monkeypatch.setattr(svc, "execute_tool", fake_execute)

    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "退货政策是什么"}]},
    )
    body = r.text
    idx = {ev: body.index(f"event: {ev}") for ev in ("conversation", "tool_call", "tool_result", "token", "done")}
    assert idx["conversation"] < idx["tool_call"] < idx["tool_result"] < idx["token"] < idx["done"]
    tc_line = [l for l in body.splitlines() if l.startswith("data:") and '"query_faq"' in l and '"args"' in l][0]
    assert json.loads(tc_line[len("data:"):].strip()) == {
        "id": "c1", "name": "query_faq", "args": {"keyword": "退货政策"}
    }
    tr_line = [l for l in body.splitlines() if l.startswith("data:") and '"ok"' in l][0]
    assert json.loads(tr_line[len("data:"):].strip()) == {
        "id": "c1", "name": "query_faq", "ok": True, "summary": "命中 1 条"
    }


async def test_tool_result_frame_carries_citations(client, monkeypatch):
    """ch04: query_faq 命中帧带 citations 键;前端弹窗数据链路。"""
    from app.api import routes as routes_mod
    from app.main import app
    from app.services import tool_chat_service as svc
    from app.tools.executor import ToolOutcome

    monkeypatch.setattr(routes_mod, "crud", FakeCrud())
    _override(app, routes_mod.dep_db_session, object())
    _override(app, routes_mod.dep_chat_model, ScriptedToolModel())

    cites = [{"n": 1, "chunk_id": 7, "section_path": "手册 > 节1", "question": "q", "answer": "a"}]

    async def fake_execute(name, args, tcid, ctx):
        return ToolOutcome(name, tcid, True, {"keyword": "k", "hits": [{"n": 1}], "refused": False, "note": ""},
                           "命中 1 条", citations=cites)

    monkeypatch.setattr(svc, "execute_tool", fake_execute)
    r = await client.post("/api/chat/stream",
                          json={"messages": [{"role": "user", "content": "退货政策"}]})
    tr_line = [l for l in r.text.splitlines() if l.startswith("data:") and '"citations"' in l][0]
    assert json.loads(tr_line[len("data:"):].strip())["citations"] == cites


async def test_conversation_id_zero_rejected_422(client):
    from app.main import app  # noqa: F401 —— 确保 app 已构建

    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}], "conversation_id": 0},
    )
    assert r.status_code == 422


async def test_ch01_style_request_without_conversation_id_still_200(client, monkeypatch):
    """不带 conversation_id 的 ch01 请求体依然兼容（spec §6）。"""
    from app.api import routes as routes_mod
    from app.main import app
    from app.services import persistence as pers_mod
    from tests.conftest import FakeChatModel

    fake_crud = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake_crud)
    monkeypatch.setattr(pers_mod, "crud", fake_crud)
    _override(app, routes_mod.dep_db_session, object())
    _override(app, routes_mod.dep_chat_model, FakeChatModel())

    r = await client.post(
        "/api/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 200
    assert r.text.count("event: token") == 3
    assert fake_crud.conv_calls == [(None, "demo_user")]  # 自动建一次性会话
