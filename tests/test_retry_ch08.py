"""ch08 T5:重试只给暂时性故障;write 恒单试;三类分诊;终局审计全量(spec 执行引擎节)。"""
import asyncio

from app.tools.audit import AuditRecord
from app.tools.executor import ToolContext, execute_tool
from app.tools.registry import ToolSpec

REC = []


async def _sink(rec: AuditRecord):
    REC.append(rec)


class SlowTool:
    name = "slow"; description = "慢"; args = {}
    def __init__(self, sleep): self.sleep = sleep; self.calls = 0
    async def ainvoke(self, args, config=None, **kw):
        self.calls += 1
        await asyncio.sleep(self.sleep)
        return {"done": True}


class BoomTool:
    name = "boom"; description = "炸"; args = {}
    def __init__(self, exc): self.exc = exc; self.calls = 0
    async def ainvoke(self, args, config=None, **kw):
        self.calls += 1
        raise self.exc


class Flaky:
    name = "flaky"; description = ""; args = {}
    def __init__(self): self.calls = 0
    async def ainvoke(self, a, config=None, **k):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("jitter")
        return {"ok": 1}


async def test_transient_retried_then_success():
    out = await execute_tool(ToolSpec(Flaky(), "readonly", "builtin"), {}, "c1",
                             ToolContext(timeout_seconds=1, max_retries=1, audit_sink=_sink))
    assert out.ok and REC[-1].status == "成功" and REC[-1].retry_count == 1


async def test_timeout_status_audited_with_duration():
    out = await execute_tool(ToolSpec(SlowTool(0.05), "readonly", "builtin"), {}, "c2",
                             ToolContext(timeout_seconds=0.01, max_retries=1, audit_sink=_sink))
    assert not out.ok and REC[-1].status == "超时" and REC[-1].retry_count == 1
    assert REC[-1].duration_ms is not None and REC[-1].duration_ms >= 0
    assert "超时" in out.result["error"]


async def test_business_error_not_retried():
    t = BoomTool(ValueError("业务硬错"))
    await execute_tool(ToolSpec(t, "readonly", "mcp", "logistics"), {}, "c3",
                       ToolContext(timeout_seconds=1, max_retries=3, audit_sink=_sink))
    assert t.calls == 1 and REC[-1].status == "失败" and REC[-1].retry_count == 0


async def test_write_never_auto_retried():
    t = BoomTool(ConnectionError("抖动也不试"))
    out = await execute_tool(ToolSpec(t, "write", "builtin"), {"x": 1}, "c4",
                             ToolContext(ticket_confirmed=True, max_retries=3, audit_sink=_sink))
    assert t.calls == 1 and "执行失败" in out.result["error"]   # 需求4:重复执行比失败更糟


async def test_write_timeout_gets_manual_check_hint():
    out = await execute_tool(ToolSpec(SlowTool(0.05), "write", "builtin"), {}, "c5",
                             ToolContext(ticket_confirmed=True, timeout_seconds=0.01,
                                         max_retries=3, audit_sink=_sink))
    assert not out.ok and "人工核实" in out.result["error"]
    assert REC[-1].status == "超时"
