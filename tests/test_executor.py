"""ch08 T3 起:execute_tool 收 ToolSpec(查找职责在调用方),FakeTool 直传包 spec。

旧 monkeypatch get_tool 模式作废;「未注册的工具」分支随之外移
(react 幻觉拒绝测在 test_react_node_ch05,legacy 面在 service/naive 各自测)。
"""

import asyncio

import pytest

from app.tools import executor as ex
from app.tools.executor import ToolContext, execute_tool
from app.tools.registry import ToolSpec


class FakeTool:
    """替身：行为由注入的 async 函数决定，记录调用次数与 config。"""

    def __init__(self, name, behavior):
        self.name = name
        self.description = ""
        self.args = {}
        self.behavior = behavior
        self.calls = 0
        self.seen_config = None
        self.seen_args = None

    async def ainvoke(self, args, config=None, **kwargs):
        self.calls += 1
        self.seen_args = args
        self.seen_config = config
        return await self.behavior(self.calls, args)


def _spec(fake):
    return ToolSpec(fake, "readonly", "builtin")


@pytest.fixture
def ctx():
    return ToolContext(conversation_id=42, timeout_seconds=0.05, max_retries=1)


async def test_success_passes_configurable(ctx):
    async def ok(call, args):
        return {"order_id": args["order_id"], "status": "运输中", "amount": 99.0}

    fake = FakeTool("query_order", ok)
    out = await execute_tool(_spec(fake), {"order_id": "1001"}, "call_1", ctx)
    assert out.ok is True
    assert out.name == "query_order" and out.tool_call_id == "call_1"
    assert out.result["status"] == "运输中"
    assert fake.seen_config["configurable"]["conversation_id"] == 42
    assert fake.seen_args == {"order_id": "1001"}
    assert "运输中" in out.summary


async def test_timeout_retries_then_error(ctx):
    async def slow(call, args):
        await asyncio.sleep(1)
        return {}

    fake = FakeTool("query_order", slow)
    out = await execute_tool(_spec(fake), {}, "call_1", ctx)
    assert out.ok is False
    assert fake.calls == 2  # 首次 + 重试 1 次
    assert "超时" in out.result["error"]


async def test_transient_error_then_success(ctx):
    # ch08 T5 分诊:暂时性=白名单(ConnectionError 族),原 RuntimeError 不再重试
    async def flaky(call, args):
        if call == 1:
            raise ConnectionError("boom")
        return {"keyword": "x", "hits": []}

    fake = FakeTool("query_faq", flaky)
    out = await execute_tool(_spec(fake), {"keyword": "x"}, "call_1", ctx)
    assert out.ok is True and fake.calls == 2
    assert out.summary == "未命中"


async def test_permanent_error_not_retried(ctx):
    # ch08 T5 三类分诊:业务硬错(ValueError 不在白名单)单试即收,不烧重试
    async def always_fail(call, args):
        raise ValueError("bad args")

    fake = FakeTool("query_faq", always_fail)
    out = await execute_tool(_spec(fake), {}, "call_1", ctx)
    assert out.ok is False and fake.calls == 1
    assert "bad args" in out.result["error"]


async def test_string_result_coerced(ctx):
    """工具若返回字符串（部分 LangChain 版本对 dict 会转 str），executor 兜底还原。"""
    async def returns_str(call, args):
        return '{"hits": [], "keyword": "y"}'

    fake = FakeTool("query_faq", returns_str)
    out = await execute_tool(_spec(fake), {}, "call_1", ctx)
    assert out.ok and out.result == {"hits": [], "keyword": "y"}


def test_make_summary_shapes():
    assert ex.make_summary("query_faq", {"keyword": "k", "hits": []}) == "未命中"
    assert ex.make_summary("query_faq", {"keyword": "k", "hits": [{"question": "q"}]}) == "命中 1 条"
    s = ex.make_summary("query_order", {"order_id": "1", "status": "派送中", "items": []})
    assert "派送中" in s
    t = ex.make_summary("create_ticket", {"ticket_no": "T20260920001", "status": "待处理"})
    assert t == "工单 T20260920001 已创建"
    e = ex.make_summary("query_order", {"error": "工具执行失败: boom"})
    assert e.startswith("失败")
    long = ex.make_summary("query_order", {"blob": "很" * 200})
    assert len(long) <= 80
