"""ch08 Task 11: 端到端验收钉 C4–C6 + C2 近似(spec「验收映射」,单元级 Fake 面)。

六条验收的可自动化项在此钉死;验收1/2/3(只注册即用/MCP 查物流/Server 加工具
客服侧不动)真 Server 面归集成测试(T6)+ README P9 手工演示脚本——本文件用
替身件不碰真 MCP/真模型。ScriptModel/fake_ticket/recorder 逐字沿用
test_confirm_flow_ch08.py(T7)的 Fake 接线。
  C4 验收4:聊天建单 → ticket_preview 暂停帧 → resume confirm →
     tickets 表 FakeCrud 恰一次 + 续播答案带工单号 token + 成功审计恰一条
  C5 验收5:resume cancel → 不建单 + 审计「权限拒绝」恰一条(含取消原因)
  C6 验收6:读超时 timeout=0.01/max_retries=2 → 审计 retry_count=2/状态「超时」/
     duration_ms 非空(实际尝试 3 次);写超时不自动重试恒 retry_count=0
  C2 近似(验收3 机内面):每轮快照含 mcp 源件被 bind_tools(内置同场)
"""
import asyncio
import contextlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.core.config import Settings
from app.tools import definitions as defs
from app.tools import executor as ex_mod
from app.tools import registry
from app.tools.audit import AuditRecord
from app.tools.executor import ToolContext, execute_tool
from app.tools.registry import BUILTIN_SPECS, ToolSpec
from app.workflows import graph as graph_mod
from app.workflows.graph import stream_graph_turn

TICKET_NO = "T20260930101"


def _settings():
    # 双 MCP 死端口 → snapshot 逐连降级到内置面(含 create_ticket),无子进程依赖
    return Settings(openai_base_url="x", openai_api_key="x", model_name="x",
                    mcp_logistics_url="http://127.0.0.1:9599/mcp",
                    mcp_aftersale_url="http://127.0.0.1:9598/mcp")


class ScriptModel:
    """意图 ainvoke 固定 JSON;ReAct astream 按轮脚本;bind_tools 记录绑定名单。"""

    def __init__(self, intent="物流", turns=()):
        self.intent = intent
        self.turns = list(turns)
        self.bound = []

    def bind_tools(self, tools, **kwargs):
        self.bound = [getattr(t, "name", str(t)) for t in tools]
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

    async def sink(record: AuditRecord):
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
        [SimpleNamespace(content="帮我建个工单，猫粮到货破损了")], _settings(), m,
        conversation_id=7, persister=None, ctx_store=None)]
    assert [k for k, _ in frames][-1] == "ticket_preview"
    return frames


# ---- C4 验收4:建单全流 confirm 落单带工单号 -----------------------------------------

async def test_c4_full_flow_confirm_lands_ticket_no(reset_graph_state, recorder,
                                                    fake_ticket):
    frames = await _reach_preview()
    payload = frames[-1][1]
    assert payload["conversation_id"] == 7 and payload["tool_call_id"] == "t1"
    rec = ScriptModel()
    resume = [f async for f in stream_graph_turn(
        None, _settings(), rec, conversation_id=7, resume_value="confirm")]
    text = "".join(p for k, p in resume if k == "token")
    assert "已为您创建工单" in text and TICKET_NO in text     # 工单号回给用户(验收4 尾断)
    assert len(fake_ticket) == 1 and fake_ticket[0]["cid"] == 7
    ok_audits = [r for r in recorder if r.status == "成功"]
    assert len(ok_audits) == 1 and ok_audits[0].tool_name == "create_ticket"
    assert [k for k, _ in resume][-1] != "ticket_preview"     # 暂停态已被消费


# ---- C5 验收5:取消=不建单+审计「权限拒绝」 -------------------------------------------

async def test_c5_cancel_no_ticket_permission_denied_audit(reset_graph_state,
                                                           recorder, fake_ticket):
    await _reach_preview()
    recorder.clear()
    resume = [f async for f in stream_graph_turn(
        None, _settings(), ScriptModel(), conversation_id=7, resume_value="cancel")]
    assert fake_ticket == []                                  # tickets 表不增行
    assert len(recorder) == 1 and recorder[0].status == "权限拒绝"
    assert recorder[0].tool_name == "create_ticket"
    assert "取消" in (recorder[0].error_message or "")
    text = "".join(p for k, p in resume if k == "token")
    assert "取消" in text


# ---- C6 验收6:超时审计列齐(读重试/写恒单试) ------------------------------------------

class SlowTool:
    name = "slow"; description = "慢"; args = {}

    def __init__(self, sleep):
        self.sleep = sleep
        self.calls = 0

    async def ainvoke(self, args, config=None, **kw):
        self.calls += 1
        await asyncio.sleep(self.sleep)
        return {"done": True}


async def _sink(rec: AuditRecord):
    REC.append(rec)


REC = []


async def test_c6_read_timeout_retries_audited_fully():
    REC.clear()
    t = SlowTool(0.05)
    out = await execute_tool(ToolSpec(t, "readonly", "mcp", "logistics"), {}, 1,
                             ToolContext(timeout_seconds=0.01, max_retries=2,
                                         audit_sink=_sink))
    assert not out.ok and "超时" in out.result["error"]
    assert t.calls == 3                                       # 首试+重试2
    r = REC[-1]
    assert r.status == "超时" and r.retry_count == 2
    assert r.duration_ms is not None and r.duration_ms >= 0   # 耗时非空(验收6 列面)


async def test_c6_write_timeout_never_auto_retried():
    REC.clear()
    t = SlowTool(0.05)
    out = await execute_tool(ToolSpec(t, "write", "builtin"), {}, 2,
                             ToolContext(ticket_confirmed=True, timeout_seconds=0.01,
                                         max_retries=2, audit_sink=_sink))
    assert not out.ok and "人工核实" in out.result["error"]
    assert t.calls == 1 and REC[-1].retry_count == 0          # 写超时恒 0 重试(需求4)
    assert REC[-1].status == "超时"


# ---- C2 近似(验收3 机内面):快照含 mcp 源件被 bind -----------------------------------

async def test_c2_mcp_sourced_spec_is_bound(reset_graph_state, monkeypatch):
    mcp_tool = SimpleNamespace(name="query_logistics", description="经 MCP 查物流", args={})

    async def fake_snapshot(settings, client=None):
        return {**BUILTIN_SPECS,
                "query_logistics": ToolSpec(mcp_tool, "readonly", "mcp", "logistics")}

    # nodes.py agent_node 函数体内 from-import 按调用时取属性 → patch 模块属性生效
    monkeypatch.setattr(registry, "snapshot_tools", fake_snapshot)
    m = ScriptModel(intent="物流", turns=[[AIMessageChunk(content="在路上了。")]])
    frames = [f async for f in stream_graph_turn(
        [SimpleNamespace(content="订单 1001 的物流到哪了")], _settings(), m,
        conversation_id=8, persister=None, ctx_store=None)]
    assert [k for k, _ in frames] and "error" not in [k for k, _ in frames]
    assert "query_logistics" in m.bound                       # mcp 源件进绑定面
    assert "create_ticket" in m.bound                         # 内置同场共存
