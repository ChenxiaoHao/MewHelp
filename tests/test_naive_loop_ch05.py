"""ch05 Task 1: 裸 Agent 循环教学对照（祛魅热身）。

fake model 脚本化 AIMessage；execute_tool 走真实执行器，ch08 T3 起
monkeypatch 本模块消费面的 BUILTIN_SPECS 注入假 ToolSpec。
"""

from langchain_core.messages import AIMessage, HumanMessage

from app.tools.executor import ToolContext
from app.tools.registry import ToolSpec
from app.workflows import naive_agent_loop as nl
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
    def __init__(self, payload, name=""):
        self.payload = payload
        self.name = name
        self.description = ""
        self.args = {}
        self.calls = 0

    async def ainvoke(self, args, config=None, **kwargs):
        self.calls += 1
        return dict(self.payload)


def _tc(name, args, id_):
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


def _install_fake_tools(monkeypatch):
    fakes = {
        "query_order": FakeTool({"order_id": "1001", "status": "运输中"}, name="query_order"),
        "query_product": FakeTool({"product_id": "2001", "name": "冻干鸡肉猫粮 2kg"}, name="query_product"),
    }
    specs = {n: ToolSpec(t, "readonly", "builtin") for n, t in fakes.items()}
    monkeypatch.setattr(nl, "BUILTIN_SPECS", specs)
    return fakes


async def test_two_step_loop_feeds_results_back(monkeypatch):
    # Ruling R5：冒烟实测流式一次并行给 2 个 tool_calls，循环须逐个执行——
    # 本测试第一轮放两个调用，钉死「多调用同轮全执行」语义。
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            _tc("query_order", {"order_id": "1001"}, "c1"),
            _tc("query_product", {"product_id": "2001"}, "c2"),
        ]),
        AIMessage(content="物流到了，已签收。"),
    ])
    fakes = _install_fake_tools(monkeypatch)
    ctx = ToolContext(conversation_id=42, timeout_seconds=5.0, max_retries=1)
    messages = [HumanMessage("订单1001物流到哪")]

    res = await naive_agent_turn(model, messages, ctx=ctx)

    assert res.steps == 1  # 一轮工具迭代
    assert [t.name for t in res.tool_calls] == ["query_order", "query_product"]
    assert fakes["query_order"].calls == 1 and fakes["query_product"].calls == 1
    assert "到了" in res.text
    # 结果以 ToolMessage 喂回：Human, AI1, Tool, Tool, AI2
    assert [type(m).__name__ for m in messages[:5]] == [
        "HumanMessage", "AIMessage", "ToolMessage", "ToolMessage", "AIMessage",
    ]
    assert model.bound_with and len(model.bound_with) == 4  # ch08 内置四件套原样 bind


async def test_no_tool_call_converges_immediately(monkeypatch):
    model = ScriptedModel([AIMessage(content="你好呀")])
    ctx = ToolContext(conversation_id=1, timeout_seconds=5.0, max_retries=0)
    res = await naive_agent_turn(model, [HumanMessage("你好")], ctx=ctx)
    assert res.steps == 0 and res.tool_calls == [] and res.text == "你好呀"
    assert model.calls == 1


async def test_exhaustion_does_not_leak_tool_json(monkeypatch):
    """M1-I2:超限收尾文本=最后一段模型自述,不得是内部工具 JSON(用户可见面)。"""
    forever = [
        AIMessage(content=f"第{i}步：继续查", tool_calls=[_tc("query_order", {"order_id": "1001"}, f"c{i}")])
        for i in range(3)
    ]
    model = ScriptedModel(forever)
    _install_fake_tools(monkeypatch)
    ctx = ToolContext(conversation_id=1, timeout_seconds=5.0, max_retries=0)
    res = await naive_agent_turn(model, [HumanMessage("查单")], ctx=ctx, max_iters=3)
    assert "order_id" not in res.text and '"status"' not in res.text
    assert "最大轮数" in res.text and "第2步" in res.text


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
