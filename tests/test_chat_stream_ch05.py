"""ch05 Task 6: SSE 接线全链测试（Graph 替换 chat_stream 编排）。

红线:token/done/error 帧字节面与 ch04 逐字符一致(断言样式复用);
RF3=无 MySQL 降级跑完一轮且不发 conversation 帧;RF4=suggestions 帧在末个
token 之后、done 之前。落库语义不动:user 行照旧走 bootstrap,
assistant/tool 行照旧走 DBChatPersister 三挂点。
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.rag import retriever as retriever_mod
from app.rag.retriever import RetrieveResult, ScoredRow
from app.services import refusals
from app.workflows import graph as graph_mod


@pytest.fixture(autouse=True)
def fresh_checkpointer():
    """生产共享 checkpointer 会让 conv-7 等线程跨测累积,每测重置。"""
    graph_mod.reset_checkpointer()
    yield
    graph_mod.reset_checkpointer()


def _override(app, dep, value):
    app.dependency_overrides[dep] = lambda: value
    return app


class GraphFakeModel:
    """意图节点 ainvoke 回固定 JSON;ReAct astream 按轮脚本吐 chunk。"""

    def __init__(self, intent="商品咨询", turns=()):
        self.intent = intent
        self.turns = list(turns)
        self.intent_calls = 0
        self.react_calls = 0
        self.last_react_msgs = None

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, msgs, **kwargs):
        self.intent_calls += 1
        return AIMessage(content=json.dumps({"intent": self.intent, "confidence": 0.9},
                                            ensure_ascii=False))

    async def astream(self, msgs, **kwargs):
        self.react_calls += 1
        self.last_react_msgs = list(msgs)
        if not self.turns:
            yield AIMessageChunk(content="（脚本已尽）")
            return
        for c in self.turns.pop(0):
            yield c


def _tool_chunk(name, args, id_):
    return AIMessageChunk(content="", tool_call_chunks=[
        {"name": name, "args": json.dumps(args, ensure_ascii=False),
         "id": id_, "type": "tool_call", "index": 0}])


def _sse(body, event):
    """按 SSE 帧序提取 data 原文列表(ch01 惯例:data=JSON,断言面用 loads 解码)。"""
    frames, cur_ev = [], None
    for line in body.splitlines():
        if line.startswith("event:"):
            cur_ev = line[len("event:"):].strip()
        elif line.startswith("data:") and cur_ev:
            frames.append((cur_ev, line[len("data:"):].strip()))
    return [d for ev, d in frames if ev == event]


def _sse_texts(body, event):
    out = []
    for d in _sse(body, event):
        try:
            v = json.loads(d)
        except json.JSONDecodeError:
            v = d
        out.append(v if isinstance(v, str) else d)
    return out


class FakeConv:
    id = 7


class FakeCrud:
    def __init__(self):
        self.messages = []
        self.conv_calls = []

    async def create_or_get_conversation(self, session, conversation_id, user_id):
        self.conv_calls.append((conversation_id, user_id))
        return FakeConv()

    async def add_message(self, session, conversation_id, role, content=None,
                          tool_calls=None, tool_call_id=None):
        self.messages.append((conversation_id, role, content, tool_calls, tool_call_id))


@pytest.fixture
def db_env(client, monkeypatch):
    """假库会话:bootstrap 走 FakeCrud,persister 写行也录进 fake_crud.messages。"""
    from app.api import routes as routes_mod
    from app.main import app
    from app.services import persistence as pers_mod

    fake_crud = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake_crud)
    monkeypatch.setattr(pers_mod, "crud", fake_crud)
    _override(app, routes_mod.dep_db_session, object())
    return fake_crud


def _post_model(client, model):
    from app.api.routes import dep_chat_model
    from app.main import app
    _override(app, dep_chat_model, model)


async def test_chitchat_frame_bytes_and_zero_model_calls(client, db_env):
    """闲聊出口:单 token 帧字节面=JSON 引号字符串(ch01 同款),零模型调用,无 suggestions。"""
    from app.main import app
    from app.workflows.nodes import CHITCHAT_FIXED

    model = GraphFakeModel(turns=[[AIMessageChunk(content="绝不该被调")]])
    _post_model(client, model)
    r = await client.post("/api/chat/stream",
                          json={"messages": [{"role": "user", "content": "你好"}]})
    assert r.status_code == 200
    assert _sse_texts(r.text, "token") == [CHITCHAT_FIXED]
    assert "event: suggestions" not in r.text
    assert "[DONE]" in r.text
    assert model.intent_calls == 0 and model.react_calls == 0
    # 落库语义:assistant 行=固定话术(persister 挂点在编排层)
    assert any(m[1] == "assistant" and m[2] == CHITCHAT_FIXED for m in db_env.messages)
    app.dependency_overrides.clear()


async def test_complaint_suggestions_after_last_token_before_done(client, db_env):
    """RF4:投诉出口 suggestions 帧位于末 token 后、done 前;items 形状钉死。"""
    model = GraphFakeModel(intent="投诉")
    _post_model(client, model)
    r = await client.post("/api/chat/stream",
                          json={"messages": [{"role": "user", "content": "我要投诉"}]})
    body = r.text
    assert body.index("event: token") < body.index("event: suggestions") \
        < body.index("event: done")
    items = json.loads(_sse(body, "suggestions")[0])["items"]
    assert [i["action"] for i in items] == ["transfer_human", "create_ticket"]
    assert all(i["label"] for i in items)
    assert model.react_calls == 0  # 投诉不进 Agent


async def test_gate_refusal_turn_streams_refusal_and_pools(client, db_env, monkeypatch):
    """知识路弱证据:检索→闸拒→单 token=兜底话术+仅转人工建议;落池 source=ch05_gate。"""
    async def fake_retrieve(query, *, strategy=None, category=None, settings=None,
                            understood=None):
        row = SimpleNamespace(category="退款", questions="q", answer="a")
        return RetrieveResult(chunks=[ScoredRow(chunk_id=1, score=0.05, row=row)])
    monkeypatch.setattr(retriever_mod, "retrieve", fake_retrieve)
    pools = []
    async def rec_pool(cid, q, src, reason, retrieved_chunks=None):
        pools.append({"cid": cid, "q": q, "src": src})
    monkeypatch.setattr(refusals, "pool_low_confidence", rec_pool)

    model = GraphFakeModel(intent="商品咨询")  # ch06 T2:退款退货已改道 refund,知识探针换商品咨询
    _post_model(client, model)
    r = await client.post("/api/chat/stream",
                          json={"messages": [{"role": "user", "content": "退款政策是什么"}]})
    body = r.text
    assert _sse_texts(body, "token") == [refusals.REFUSAL_ANSWER]
    items = json.loads(_sse(body, "suggestions")[0])["items"]
    assert [i["action"] for i in items] == ["transfer_human"]
    assert body.index("event: suggestions") < body.index("event: done")
    assert pools == [{"cid": 7, "q": "退款政策是什么", "src": "ch05_gate"}]
    assert model.react_calls == 0


async def test_data_route_tool_frames_order_and_persistence(client, db_env, monkeypatch):
    """业务路全链:tool_call→tool_result→token 帧序;persister 三挂点行齐
    (assistant+tool_calls / tool / assistant 终答)——落库语义不动的图路径证明。"""
    from app.agents import react as react_mod
    from app.tools.executor import ToolOutcome

    async def fake_execute(spec, args, tcid, ctx):
        assert ctx.conversation_id == 7  # 会话 id 经 config 注入工具上下文
        return ToolOutcome(spec.name, tcid, True, {"order_id": "1001"}, "运输中")
    monkeypatch.setattr(react_mod, "execute_tool", fake_execute)

    model = GraphFakeModel(intent="物流", turns=[
        [_tool_chunk("query_order", {"order_id": "1001"}, "c1")],
        [AIMessageChunk(content="包裹"), AIMessageChunk(content="运输中")],
    ])
    _post_model(client, model)
    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "订单1001的物流到哪了"}],
              "conversation_id": 7})
    body = r.text
    idx = {ev: body.index(f"event: {ev}") for ev in
           ("conversation", "tool_call", "tool_result", "token", "done")}
    assert idx["conversation"] < idx["tool_call"] < idx["tool_result"] < idx["token"] < idx["done"]
    assert json.loads(_sse(body, "tool_call")[0]) == {
        "id": "c1", "name": "query_order", "args": {"order_id": "1001"}}
    assert json.loads(_sse(body, "tool_result")[0]) == {
        "id": "c1", "name": "query_order", "ok": True, "summary": "运输中"}
    assert "".join(json.loads(t) for t in _sse(body, "token")) == "包裹运输中"
    rows = db_env.messages
    assert (7, "user", "订单1001的物流到哪了", None, None) in rows          # bootstrap
    asst_tc = [m for m in rows if m[1] == "assistant" and m[3]]
    assert asst_tc and asst_tc[0][3] == [
        {"id": "c1", "name": "query_order", "args": {"order_id": "1001"}}]  # on_tool_calls
    assert any(m[1] == "tool" and m[4] == "c1" for m in rows)               # on_tool_result
    assert any(m[1] == "assistant" and m[2] == "包裹运输中" for m in rows)   # on_final_answer
    assert "event: suggestions" not in body  # Agent 正常收敛不发建议


async def test_no_db_degrades_full_turn(client, monkeypatch):
    """RF3:引擎未初始化(session=None)→ 无 conversation 帧、无落库,一轮照样跑完。"""
    from app.api import routes as routes_mod
    from app.main import app
    from app.workflows.nodes import CHITCHAT_FIXED

    async def none_session():
        yield None
    app.dependency_overrides[routes_mod.dep_db_session] = none_session
    model = GraphFakeModel()
    _post_model(client, model)
    r = await client.post("/api/chat/stream",
                          json={"messages": [{"role": "user", "content": "你好"}]})
    assert r.status_code == 200
    assert "event: conversation" not in r.text
    assert _sse_texts(r.text, "token") == [CHITCHAT_FIXED]
    assert "[DONE]" in r.text
    app.dependency_overrides.clear()


async def test_error_frame_on_upstream_failure(client, db_env):
    """error 帧字节面回归:上游炸→event:error + {"detail"} 帧,关流不发 done。"""
    from app.main import app

    class Boom(GraphFakeModel):
        # ch06 T2 重定向:意图节点异常已被 _judge 降级层吞掉(归「其他」不再抛穿),
        # error 帧注入点移到 Agent 流式出口(data 路线,不触真检索)。
        async def astream(self, msgs, **kwargs):
            raise RuntimeError("upstream down")
            yield  # pragma: no cover —— 保持 asyncgen 签名

    _post_model(client, Boom(intent="物流"))
    r = await client.post("/api/chat/stream",
                          json={"messages": [{"role": "user", "content": "查个单"}]})
    err = _sse(r.text, "error")
    assert err and "upstream down" in json.loads(err[0])["detail"]
    assert "[DONE]" not in r.text
    app.dependency_overrides.clear()
