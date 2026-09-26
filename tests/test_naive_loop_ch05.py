"""ch05 Task 1: 裸 Agent 循环教学对照（祛魅热身）。

fake model 脚本化 AIMessage；execute_tool 走真实执行器，仅 monkeypatch
get_tool 注入假工具（对齐 tests/test_executor.py 的既有模式）。
"""

from langchain_core.messages import AIMessage, HumanMessage

from app.tools import executor as ex
from app.tools.executor import ToolContext
from app.workflows.naive_agent_loop import naive_agent_turn


class ScriptedModel:
    """按脚本吐 AIMessage；bind_tools 返回 self，并记录调用次数。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.bound_with = None

    def bind_tools(self, tools):
        self.bound_with = tools
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return self.script.pop(0)


class FakeTool:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def ainvoke(self, args, config=None, **kwargs):
        self.calls += 1
        return dict(self.payload)


def _tc(name, args, id_):
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


def _install_fake_tools(monkeypatch):
    fakes = {
        "query_order": FakeTool({"order_id": "1001", "status": "运输中"}),
        "query_logistics": FakeTool({"order_id": "1001", "status": "已签收"}),
    }
    monkeypatch.setattr(ex, "get_tool", lambda name: fakes.get(name))
    return fakes


async def test_two_step_loop_feeds_results_back(monkeypatch):
    # Ruling R5：冒烟实测流式一次并行给 2 个 tool_calls，循环须逐个执行——
    # 本测试第一轮放两个调用，钉死「多调用同轮全执行」语义。
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            _tc("query_order", {"order_id": "1001"}, "c1"),
            _tc("query_logistics", {"order_id": "1001"}, "c2"),
        ]),
        AIMessage(content="物流到了，已签收。"),
    ])
    fakes = _install_fake_tools(monkeypatch)
    ctx = ToolContext(conversation_id=42, timeout_seconds=5.0, max_retries=1)
    messages = [HumanMessage("订单1001物流到哪")]

    res = await naive_agent_turn(model, messages, ctx=ctx)

    assert res.steps == 1  # 一轮工具迭代
    assert [t.name for t in res.tool_calls] == ["query_order", "query_logistics"]
    assert fakes["query_order"].calls == 1 and fakes["query_logistics"].calls == 1
    assert "到了" in res.text
    # 结果以 ToolMessage 喂回：Human, AI1, Tool, Tool, AI2
    assert [type(m).__name__ for m in messages[:5]] == [
        "HumanMessage", "AIMessage", "ToolMessage", "ToolMessage", "AIMessage",
    ]
    assert model.bound_with and len(model.bound_with) == 5  # ch02 注册表原样 bind


async def test_no_tool_call_converges_immediately(monkeypatch):
    model = ScriptedModel([AIMessage(content="你好呀")])
    ctx = ToolContext(conversation_id=1, timeout_seconds=5.0, max_retries=0)
    res = await naive_agent_turn(model, [HumanMessage("你好")], ctx=ctx)
    assert res.steps == 0 and res.tool_calls == [] and res.text == "你好呀"
    assert model.calls == 1


async def test_hits_max_iters_without_convergence(monkeypatch):
    # Review Focus 2 前置：模型永不收敛 → 循环必须在 max_iters 内返回而非死循环
    forever = [
        AIMessage(content="", tool_calls=[_tc("query_order", {"order_id": "x"}, f"c{i}")])
        for i in range(3)
    ]
    model = ScriptedModel(forever)
    _install_fake_tools(monkeypatch)
    ctx = ToolContext(conversation_id=1, timeout_seconds=5.0, max_retries=0)
    res = await naive_agent_turn(
        model, [HumanMessage("查单")], ctx=ctx, max_iters=3,
    )
    assert res.steps == 3  # 示例化断言：以 max_iters 为准
    assert "最大轮数" in res.text
