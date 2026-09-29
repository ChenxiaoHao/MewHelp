"""ch08 T7:建单确认流图侧(spec 确认流节)。

脚本模型第一轮调 create_ticket(必填齐),第二轮收尾文本 → agent 捕获
ticket_request → ticket_preview 条件边 → ticket_confirm 节点 interrupt 暂停;
resume confirm/cancel 双路 + 新消息隐式 cancel drain。审计面经
monkeypatch executor.db_audit_sink 录制;写库经 fake_ticket 计数。
"""
import contextlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.core.config import Settings
from app.tools import definitions as defs
from app.tools import executor as ex_mod
from app.workflows import graph as graph_mod
from app.workflows.graph import stream_graph_turn

TICKET_NO = "T20260930001"


def _settings():
    # 双 MCP 死端口 → snapshot 逐连降级到内置面(含 create_ticket),无子进程依赖
    return Settings(openai_base_url="x", openai_api_key="x", model_name="x",
                    mcp_logistics_url="http://127.0.0.1:9599/mcp",
                    mcp_aftersale_url="http://127.0.0.1:9598/mcp")


class ScriptModel:
    """意图 ainvoke 固定 JSON;ReAct astream 按轮脚本。"""

    def __init__(self, intent="物流", turns=()):
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
    return [_tool_chunk("create_ticket",
                        {"description": "猫粮到货破损", "ticket_type": "售后"}, "t1"),
            AIMessageChunk(content="预览卡片已发出")], [AIMessageChunk(content="，请确认。")]


@pytest.fixture
def reset_graph_state():
    graph_mod.reset_checkpointer()
    yield
    graph_mod.reset_checkpointer()


@pytest.fixture
def recorder(monkeypatch):
    recs = []

    async def sink(record):
        recs.append(record)

    monkeypatch.setattr(ex_mod, "db_audit_sink", sink)
    return recs


@pytest.fixture
def fake_ticket(monkeypatch):
    """tickets 写入面替身:计数=「不增行/恰一次」断言面。"""
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


async def _reach_preview():
    m = ScriptModel(turns=_ticket_turns())
    frames = [f async for f in stream_graph_turn(
        [SimpleNamespace(content="帮我建个工单,猫粮到货破损了")], _settings(), m,
        conversation_id=7, persister=None, ctx_store=None)]
    assert [k for k, _ in frames][-1] == "ticket_preview"
    return frames


async def test_preview_frame_and_pause(reset_graph_state, recorder, fake_ticket):
    frames = await _reach_preview()
    kinds = [k for k, _ in frames]
    assert kinds[-1] == "ticket_preview"          # 暂停语义:preview 后即末帧,无 done/suggestions
    assert "suggestions" not in kinds
    payload = frames[-1][1]
    assert payload["ticket_type"] == "售后" and payload["description"] == "猫粮到货破损"
    assert payload["tool_call_id"] == "t1" and payload["conversation_id"] == 7
    # tool_result 回灌帧照发(ok=False+等待文案,模型见错误自收敛)
    tr = [p for k, p in frames if k == "tool_result"]
    assert tr and tr[-1]["ok"] is False and tr[-1]["summary"] == "等待客户确认"
    # 中间态不发审计(T4 裁决端到端面):权限闸拒绝不落账,写库零调用
    assert recorder == [] and fake_ticket == []


async def test_resume_confirm_lands_ticket(reset_graph_state, recorder, fake_ticket):
    await _reach_preview()
    frames = [f async for f in stream_graph_turn(
        None, _settings(), ScriptModel(), conversation_id=7,
        resume_value="confirm")]
    text = "".join(p for k, p in frames if k == "token")
    assert "已为您创建工单" in text and TICKET_NO in text
    assert fake_ticket[0]["desc"] == "猫粮到货破损"
    assert fake_ticket[0]["cid"] == 7 and fake_ticket[0]["type"] == "售后"
    assert [r.status for r in recorder] == ["成功"]   # 重放安全:审计恰一条
    assert [k for k, _ in frames][-1] != "ticket_preview"


async def test_resume_cancel_audits_permission_denied(reset_graph_state, recorder,
                                                      fake_ticket):
    await _reach_preview()
    recorder.clear()
    frames = [f async for f in stream_graph_turn(
        None, _settings(), ScriptModel(), conversation_id=7,
        resume_value="cancel")]
    text = "".join(p for k, p in frames if k == "token")
    assert "取消" in text
    assert fake_ticket == []                       # tickets 表不增行(计数面)
    assert len(recorder) == 1 and recorder[-1].status == "权限拒绝"
    assert recorder[-1].tool_name == "create_ticket"
    assert "取消" in (recorder[-1].error_message or "")


async def test_implicit_cancel_on_new_message(reset_graph_state, recorder, fake_ticket):
    """Review Focus 1:卡片弹出后直接发新消息 → 不 500、旧单按取消收账、新轮正常答。"""
    await _reach_preview()
    recorder.clear()
    m2 = ScriptModel(turns=[[AIMessageChunk(content="好的,发货地是江苏。")]])
    frames = [f async for f in stream_graph_turn(
        [SimpleNamespace(content="算了,先问下发货地")], _settings(), m2,
        conversation_id=7, persister=None, ctx_store=None)]
    kinds = [k for k, _ in frames]
    assert "token" in kinds and "error" not in kinds
    assert recorder[-1].status == "权限拒绝"        # 隐式 cancel 落账
    assert fake_ticket == []
    text = "".join(p for k, p in frames if k == "token")
    assert "发货地" in text                        # 新轮正常答


async def test_interrupt_replay_side_effect_free(reset_graph_state, recorder,
                                                 fake_ticket):
    """resume 重放:ticket_confirm 从头跑,interrupt 前必须零副作用——
    confirm 路径建单恰一次、成功审计恰一条、取消审计零条。"""
    await _reach_preview()
    assert fake_ticket == []                       # 首过:interrupt 前无执行
    frames = [f async for f in stream_graph_turn(
        None, _settings(), ScriptModel(), conversation_id=7, resume_value="confirm")]
    assert len(fake_ticket) == 1
    assert [r.status for r in recorder if r.status == "成功"] == ["成功"]
    assert not any(r.status == "权限拒绝" for r in recorder)
