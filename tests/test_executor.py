import asyncio

import pytest

from app.tools import executor as ex
from app.tools.executor import ToolContext, execute_tool


class FakeTool:
    """替身：行为由注入的 async 函数决定，记录调用次数与 config。"""

    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = 0
        self.seen_config = None
        self.seen_args = None

    async def ainvoke(self, args, config=None, **kwargs):
        self.calls += 1
        self.seen_args = args
        self.seen_config = config
        return await self.behavior(self.calls, args)


@pytest.fixture
def ctx():
    return ToolContext(conversation_id=42, timeout_seconds=0.05, max_retries=1)


async def test_success_passes_configurable(monkeypatch, ctx):
    async def ok(call, args):
        return {"order_id": args["order_id"], "status": "运输中", "amount": 99.0}

    fake = FakeTool(ok)
    monkeypatch.setattr(ex, "get_tool", lambda name: fake)
    out = await execute_tool("query_order", {"order_id": "1001"}, "call_1", ctx)
    assert out.ok is True
    assert out.name == "query_order" and out.tool_call_id == "call_1"
    assert out.result["status"] == "运输中"
    assert fake.seen_config["configurable"]["conversation_id"] == 42
    assert fake.seen_args == {"order_id": "1001"}
    assert "运输中" in out.summary


async def test_timeout_retries_then_error(monkeypatch, ctx):
    async def slow(call, args):
        await asyncio.sleep(1)
        return {}

    fake = FakeTool(slow)
    monkeypatch.setattr(ex, "get_tool", lambda name: fake)
    out = await execute_tool("query_order", {}, "call_1", ctx)
    assert out.ok is False
    assert fake.calls == 2  # 首次 + 重试 1 次
    assert "超时" in out.result["error"]


async def test_transient_error_then_success(monkeypatch, ctx):
    async def flaky(call, args):
        if call == 1:
            raise RuntimeError("boom")
        return {"keyword": "x", "hits": []}

    fake = FakeTool(flaky)
    monkeypatch.setattr(ex, "get_tool", lambda name: fake)
    out = await execute_tool("query_faq", {"keyword": "x"}, "call_1", ctx)
    assert out.ok is True and fake.calls == 2
    assert out.summary == "未命中"


async def test_permanent_error_exhausts_retries(monkeypatch, ctx):
    async def always_fail(call, args):
        raise ValueError("bad args")

    fake = FakeTool(always_fail)
    monkeypatch.setattr(ex, "get_tool", lambda name: fake)
    out = await execute_tool("query_faq", {}, "call_1", ctx)
    assert out.ok is False and fake.calls == 2
    assert "bad args" in out.result["error"]


async def test_unknown_tool_returns_error_outcome(ctx):
    out = await execute_tool("no_such_tool", {}, "call_1", ctx)
    assert out.ok is False
    assert "未注册" in out.result["error"]


async def test_string_result_coerced(monkeypatch, ctx):
    """工具若返回字符串（部分 LangChain 版本对 dict 会转 str），executor 兜底还原。"""
    async def returns_str(call, args):
        return '{"hits": [], "keyword": "y"}'

    fake = FakeTool(returns_str)
    monkeypatch.setattr(ex, "get_tool", lambda name: fake)
    out = await execute_tool("query_faq", {}, "call_1", ctx)
    assert out.ok and out.result == {"hits": [], "keyword": "y"}


def test_make_summary_shapes():
    assert ex.make_summary("query_faq", {"keyword": "k", "hits": []}) == "未命中"
    assert ex.make_summary("query_faq", {"keyword": "k", "hits": [{"question": "q"}]}) == "命中 1 条"
    s = ex.make_summary("query_logistics", {"order_id": "1", "carrier": "中通快递", "current_status": "派送中", "traces": []})
    assert "派送中" in s
    t = ex.make_summary("create_ticket", {"ticket_no": "T20260920001", "status": "待处理"})
    assert t == "工单 T20260920001 已创建"
    e = ex.make_summary("query_order", {"error": "工具执行失败: boom"})
    assert e.startswith("失败")
    long = ex.make_summary("query_order", {"blob": "很" * 200})
    assert len(long) <= 80
