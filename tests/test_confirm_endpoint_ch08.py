"""ch08 T8:POST /api/tickets/confirm 端点面(spec 确认流节/需求7)。

链路:chat/stream 走到 ticket_preview 暂停(同进程共享 checkpointer,线程
conv-{cid} 挂着 interrupt)→ confirm 端点 Command(resume) 续播 SSE。
红线:409=无 pending 卡;422=乱值且 interrupt 不被消耗(Review Focus 2,
随后 confirm 仍能成);帧面复用 chat_stream 的映射 helper。
"""
import contextlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.api import routes as routes_mod
from app.core.config import Settings
from app.main import app
from app.tools import definitions as defs
from app.tools import executor as ex_mod
from app.workflows import graph as graph_mod

TICKET_NO = "T20260930002"


def _settings():
    # 双 MCP 死端口 → snapshot 恒降级内置面(T7 同款防线:防手起 server 污染)
    return Settings(_env_file=None, openai_base_url="x", openai_api_key="x",
                    model_name="x", tool_timeout_seconds=5.0, tool_max_retries=0,
                    retrieval_low_conf_threshold=0.161, max_agent_steps=6,
                    react_token_budget=8000, history_token_budget=4000,
                    demo_user_id="demo_user",
                    mcp_logistics_url="http://127.0.0.1:9599/mcp",
                    mcp_aftersale_url="http://127.0.0.1:9598/mcp")


class ScriptModel:
    """意图 ainvoke 固定 JSON;ReAct astream 按轮脚本(图级 T7 同款)。"""

    def __init__(self, intent="物流", turns=()):  # 业务路旁路 retrieve(T7 同款:防打真 Milvus/embeddings)
        self.intent = intent
        self.turns = list(turns)

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, msgs, **kwargs):
        return AIMessage(content=json.dumps({"intent": self.intent, "confidence": 0.9},
                                            ensure_ascii=False))

    async def astream(self, msgs, **kwargs):
        if not self.turns:
            yield AIMessageChunk(content="（脚本已尽）")
            return
        for c in self.turns.pop(0):
            yield c


def _tool_chunk(name, args, id_):
    return AIMessageChunk(content="", tool_call_chunks=[
        {"name": name, "args": json.dumps(args, ensure_ascii=False),
         "id": id_, "type": "tool_call", "index": 0}])


def _ticket_turns():
    # 两轮形状(T7 同款):轮1=tool_call+文本,轮2=模型见「等待客户确认」自收敛收尾
    return [_tool_chunk("create_ticket",
                        {"description": "猫粮到货破损", "ticket_type": "售后"}, "tc-e1"),
            AIMessageChunk(content="已生成工单预览")], [AIMessageChunk(content="，请确认。")]


def _sse(body, event):
    frames, cur_ev = [], None
    for line in body.splitlines():
        if line.startswith("event:"):
            cur_ev = line[len("event:"):].strip()
        elif line.startswith("data:") and cur_ev:
            frames.append((cur_ev, line[len("data:"):].strip()))
    return [d for ev, d in frames if ev == event]


def _token_text(body):
    return "".join(json.loads(d) for d in _sse(body, "token"))


class FakeConv:
    id = 7


class FakeCrud:
    def __init__(self):
        self.messages = []

    async def create_or_get_conversation(self, session, conversation_id, user_id):
        return FakeConv()

    async def add_message(self, session, conversation_id, role, content=None,
                          tool_calls=None, tool_call_id=None):
        self.messages.append((conversation_id, role, content, tool_calls, tool_call_id))


@pytest.fixture(autouse=True)
def fresh_checkpointer():
    graph_mod.reset_checkpointer()
    routes_mod._confirm_inflight.clear()   # M2-I3 单入口面防测间残留
    yield
    graph_mod.reset_checkpointer()
    routes_mod._confirm_inflight.clear()


@pytest.fixture
def env(client, monkeypatch):
    """端点测试台:dep_settings=死端口面 + dep_db_session=假 session + crud 替身录制。"""
    fake_crud = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake_crud)
    from app.services import persistence as pers_mod
    monkeypatch.setattr(pers_mod, "crud", fake_crud)
    app.dependency_overrides[routes_mod.dep_settings] = lambda: _settings()
    app.dependency_overrides[routes_mod.dep_db_session] = lambda: object()
    return fake_crud


@pytest.fixture
def post_model(client):
    def _post(model):
        app.dependency_overrides[routes_mod.dep_chat_model] = lambda: model
        return model
    return _post


@pytest.fixture
def recorder(monkeypatch):
    recs = []

    async def sink(record):
        recs.append(record)

    monkeypatch.setattr(ex_mod, "db_audit_sink", sink)
    return recs


@pytest.fixture
def fake_ticket(monkeypatch):
    calls = []

    async def fake_crud(session, *, conversation_id, description, ticket_type):
        calls.append({"cid": conversation_id, "desc": description, "type": ticket_type})
        return SimpleNamespace(ticket_no=TICKET_NO, status="待处理")

    @contextlib.asynccontextmanager
    async def fake_cm():
        yield object()

    monkeypatch.setattr(defs, "crud_create_ticket", fake_crud)
    monkeypatch.setattr(defs, "get_session_factory", lambda: fake_cm)
    return calls


async def _reach_preview(client, post_model):
    post_model(ScriptModel(turns=_ticket_turns()))
    r1 = await client.post("/api/chat/stream",
                           json={"messages": [{"role": "user",
                                               "content": "帮我建个工单，猫粮到货破损了"}]})
    assert r1.status_code == 200 and "ticket_preview" in r1.text
    return r1.text


async def test_confirm_sse_streams_ticket_no(env, post_model, recorder, fake_ticket, client):
    """preview 挂起 → confirm 续播:token 帧文本含工单号,建单恰一次,成功审计恰一行。"""
    await _reach_preview(client, post_model)
    post_model(ScriptModel())   # resume 轮图不再走 react(confirm 节点出脚本话术)
    resp = await client.post("/api/tickets/confirm",
                             json={"conversation_id": 7, "decision": "confirm"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "已为您创建工单" in _token_text(resp.text) and TICKET_NO in _token_text(resp.text)
    assert "[DONE]" in resp.text
    assert len(fake_ticket) == 1 and fake_ticket[0]["cid"] == 7
    assert [r.status for r in recorder] == ["成功"]   # 重放安全:审计恰一条


async def test_cancel_endpoint(env, post_model, recorder, fake_ticket, client):
    await _reach_preview(client, post_model)
    post_model(ScriptModel())
    resp = await client.post("/api/tickets/confirm",
                             json={"conversation_id": 7, "decision": "cancel"})
    assert resp.status_code == 200
    assert "取消" in _token_text(resp.text)
    assert fake_ticket == []
    assert len(recorder) == 1 and recorder[-1].status == "权限拒绝"


async def test_no_pending_interrupt_is_409(env, client):
    resp = await client.post("/api/tickets/confirm",
                             json={"conversation_id": 999, "decision": "confirm"})
    assert resp.status_code == 409


async def test_unknown_decision_422_does_not_consume_interrupt(
        env, post_model, recorder, fake_ticket, client):
    """Review Focus 2:乱值 decision 被请求体校验挡下(不进 resume),interrupt 完好;
    随后 decision=confirm 仍能成。"""
    await _reach_preview(client, post_model)
    bad = await client.post("/api/tickets/confirm",
                            json={"conversation_id": 7, "decision": "delete"})
    assert bad.status_code == 422
    assert fake_ticket == [] and recorder == []     # interrupt 未被消耗

    post_model(ScriptModel())
    ok = await client.post("/api/tickets/confirm",
                           json={"conversation_id": 7, "decision": "confirm"})
    assert ok.status_code == 200
    assert TICKET_NO in _token_text(ok.text)


async def test_confirm_resume_assistant_row_saved(env, post_model, recorder,
                                                  fake_ticket, client):
    """M2-I2 端点面:confirm 续播答复须进 messages 表(刷新/重启回填后工单号可见)。"""
    await _reach_preview(client, post_model)
    post_model(ScriptModel())
    resp = await client.post("/api/tickets/confirm",
                             json={"conversation_id": 7, "decision": "confirm"})
    assert resp.status_code == 200
    assert any(role == "assistant" and TICKET_NO in (content or "")
               for (_cid, role, content, _tc, _tcid) in env.messages), env.messages


async def test_concurrent_double_confirm_creates_once(env, post_model, recorder,
                                                      fake_ticket, client):
    """M2-I3:409 预检+resume 非原子——卡片双击并发两发 confirm,写路径防重:
    恰一发进 resume(200+建单恰一次),另一发被挡(409)。"""
    import asyncio
    await _reach_preview(client, post_model)
    post_model(ScriptModel())
    r1, r2 = await asyncio.gather(
        client.post("/api/tickets/confirm",
                    json={"conversation_id": 7, "decision": "confirm"}),
        client.post("/api/tickets/confirm",
                    json={"conversation_id": 7, "decision": "confirm"}))
    assert sorted([r1.status_code, r2.status_code]) == [200, 409]
    assert len(fake_ticket) == 1, "唯一 write 的执行闸必须防重(需求4)"


async def test_confirm_race_new_message_single_terminal(env, post_model, recorder,
                                                        fake_ticket, client, monkeypatch):
    """终审 I-1:M2-I3 自述场景的另一半——点「确认」后秒发新消息(或双标签页):
    drain("cancel") 与 confirm resume 并发重放同线程 ticket_confirm → 双建单+
    「成功」「权限拒绝」终局并存(审计/历史自相矛盾)。per-thread 锁必须把
    「查 pending+drain」与「查 pending+resume」串进同一临界区:恰一个终局。"""
    import asyncio
    await _reach_preview(client, post_model)
    app.dependency_overrides[routes_mod.dep_chat_model] = lambda: ScriptModel()

    # 触窗放大:confirm 侧重放建单挂 0.3s(慢包裹在 fake_ticket 录制替身外层,
    # 调用照常入 calls 清单)。ch09 收编:盲 gather 在负载下会翻转为
    # 「drain 完整抢先→confirm 合法 409」的假失败——改事件定序:confirm 先起,
    # 重放真正开跑(slow_create 进门=锁已被 resume 接力持有)后才发新消息,
    # 钉死本用例要审的交错(drain 撞进重放窗口);无锁接力时双终局照样现形。
    entered = asyncio.Event()
    rec_create = defs.crud_create_ticket

    async def slow_create(session, **kw):
        entered.set()
        await asyncio.sleep(0.3)
        return await rec_create(session, **kw)
    monkeypatch.setattr(defs, "crud_create_ticket", slow_create)

    t_confirm = asyncio.create_task(client.post(
        "/api/tickets/confirm",
        json={"conversation_id": 7, "decision": "confirm"}))
    await asyncio.wait_for(entered.wait(), 5)  # 超时即失败,不再赌调度
    r_chat = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "顺便问下退货地址"}],
              "conversation_id": 7})
    r_confirm = await t_confirm
    assert r_confirm.status_code == 200 and r_chat.status_code == 200
    assert "event:error" not in r_chat.text, "RF1:新消息轮不得带 error 帧"
    assert len(fake_ticket) <= 1, "并发竞态不得双建单(需求4 唯一 write 闸)"
    statuses = [r.status for r in recorder if r.tool_name == "create_ticket"]
    assert statuses in ([], ["成功"], ["权限拒绝"]), \
        f"同 interrupt 只许一个终局审计,得 {statuses}"
    rows = [content for (_cid, role, content, _tc, _tcid) in env.messages
            if role == "assistant"]
    assert not any("已取消本次建单" in (c or "") for c in rows) or \
        TICKET_NO not in "".join(c or "" for c in rows), \
        "messages 表同线程既落取消又落工单号=自相矛盾终局(I2 回填面取证)"
