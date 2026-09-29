"""ch05 Task 5: ReAct 主力 Agent 流式节点(app/agents/react.py)。

AgentEvent 元组形状逐字符对齐 ch04 ToolEvent(routes 层零改动复用,T6):
("token", str) | ("tool_call", {id,name,args}) | ("tool_result", {id,name,ok,summary})
新增 ("done", {"steps": int, "suggestions": list})。
模型用真 AIMessageChunk 脚本化(ch04 同款 + 累加聚合路径),工具执行按仓内
既有模式 monkeypatch executor.get_tool。
"""

import json
from types import SimpleNamespace

from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from app.agents.react import react_agent_stream
from app.core.config import Settings
from app.tools import executor as ex
from app.workflows.nodes import TRANSFER_HUMAN


class FakeTool:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def ainvoke(self, args, config=None, **kwargs):
        self.calls += 1
        return dict(self.payload)


def _install_fake_tools(monkeypatch):
    fakes = {
        "query_order": FakeTool({"order_id": "1001", "status": "运输中"}),
        "query_logistics": FakeTool({"order_id": "1001", "status": "已签收"}),
    }
    monkeypatch.setattr(ex, "get_tool", lambda name: fakes.get(name))
    return fakes


def _turn(text="", tool_calls=()):
    """一个模型轮 = chunk 列表:文本 chunk 若干 + 工具调用 chunk(整段 args)。"""
    chunks = []
    if text:
        chunks.append(AIMessageChunk(content=text))
    for i, (name, args, id_) in enumerate(tool_calls):
        chunks.append(AIMessageChunk(content="", tool_call_chunks=[
            {"name": name, "args": json.dumps(args, ensure_ascii=False),
             "id": id_, "type": "tool_call", "index": i},
        ]))
    return chunks


class ScriptStreamModel:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = 0
        self.last_msgs = None
        self.bound_with = None

    def bind_tools(self, tools):
        self.bound_with = tools
        return self

    async def astream(self, messages):
        self.calls += 1
        self.last_msgs = list(messages)
        for c in self.turns.pop(0):
            yield c


def st(**over):
    base = dict(tool_timeout_seconds=5.0, tool_max_retries=0,
                max_agent_steps=6, react_token_budget=8000)
    base.update(over)
    return SimpleNamespace(**base)


def state_with_evidence():
    return {"messages": [HumanMessage(content="订单1001的物流到哪了")],
            "resolved_query": "订单1001的物流到哪了",
            "evidence": [{"chunk_id": 1, "score": 0.9, "text": "类目\n问\n答"}],
            "conversation_id": 42}


async def test_complex_question_walks_multiple_steps(monkeypatch):
    """验收 5 单测面:ReAct >1 步。"""
    _install_fake_tools(monkeypatch)
    model = ScriptStreamModel([
        _turn("先看订单。", [("query_order", {"order_id": "1001"}, "c1")]),
        _turn(tool_calls=[("query_logistics", {"order_id": "1001"}, "c2")]),
        _turn("已签收，放在驿站。"),
    ])
    events = [e async for e in react_agent_stream(state_with_evidence(), st(), model)]
    assert sum(1 for k, _ in events if k == "tool_call") >= 2
    assert events[-1][0] == "done" and events[-1][1]["steps"] >= 2
    assert "".join(d for k, d in events if k == "token").endswith("已签收，放在驿站。")


async def test_clarifying_question_only_tokens_and_zero_steps(monkeypatch):
    """缺信息追问:无 tool_calls 的提问 → 事件流只有 token+done,steps==0。"""
    model = ScriptStreamModel([_turn("请问您的订单号是多少？")])
    events = [e async for e in react_agent_stream(state_with_evidence(), st(), model)]
    assert [k for k, _ in events] == ["token", "done"]
    assert events[-1][1] == {"steps": 0, "suggestions": []}


async def test_rf2_forever_tool_calls_terminate_within_max_iters(monkeypatch):
    """Review Focus 2:模型恒要工具 → max_agent_steps 内收流不抛(T8 接管轮数上限)。"""
    _install_fake_tools(monkeypatch)
    forever = [_turn(tool_calls=[("query_order", {"order_id": "x"}, f"c{i}")])
               for i in range(5)]
    model = ScriptStreamModel(forever)
    events = [e async for e in react_agent_stream(
        state_with_evidence(), st(max_agent_steps=5), model)]
    assert events[-1][0] == "done" and events[-1][1]["steps"] == 5
    assert events[-1][1]["suggestions"] == [TRANSFER_HUMAN]  # 超限=兜底建议
    assert model.calls == 5


async def test_token_budget_breaker_streams_no_more(monkeypatch):
    """每轮 token 累计对 react_token_budget 熔断:超预算当轮收尾,不再进循环。
    T8 计数同源:CJK 估算器(40 汉字≈40+4=44)替代 count_tokens_approximately
    (旧口径≈10 不熔断)——阈值按新源重校,旧实现必显形(第二轮照跑)。"""
    _install_fake_tools(monkeypatch)
    long_text = "很" * 40
    model = ScriptStreamModel([
        _turn(long_text, [("query_order", {"order_id": "1001"}, "c1")]),
        _turn("旧计数口径才会走到这第二轮"),
    ])
    events = [e async for e in react_agent_stream(
        state_with_evidence(), st(react_token_budget=40), model)]
    assert events[-1][1]["suggestions"] == [TRANSFER_HUMAN]
    assert model.calls == 1  # 熔断后无第二轮调用


async def test_event_payload_shapes_align_tool_event(monkeypatch):
    """T6 零改动前提:tool_call/tool_result 帧与 ch04 ToolEvent 键逐一对齐。"""
    _install_fake_tools(monkeypatch)
    model = ScriptStreamModel([
        _turn(tool_calls=[("query_order", {"order_id": "1001"}, "c1")]),
        _turn("运输中。"),
    ])
    events = [e async for e in react_agent_stream(state_with_evidence(), st(), model)]
    call = next(d for k, d in events if k == "tool_call")
    result = next(d for k, d in events if k == "tool_result")
    assert set(call) == {"id", "name", "args"}
    assert set(result) == {"id", "name", "ok", "summary"}


async def test_state_evidence_never_prepends_system_message(monkeypatch):
    """ch07 T8 改形(规4 旧断言翻转):证据/订单注入=装配段5 之责(当前句后一条
    Human);react 不再读 state.evidence/order_data,任何 System 前置都不许再出现。"""
    state = state_with_evidence()
    state["order_data"] = {"order_id": "1001"}
    model = ScriptStreamModel([_turn("答案")])
    [e async for e in react_agent_stream(state, st(), model)]
    assert not any(isinstance(m, SystemMessage) for m in model.last_msgs)
    assert isinstance(model.last_msgs[0], HumanMessage)   # 入参 msgs 原样透传


def test_settings_react_defaults_pinned():
    """P3 拍板默认值钉死(T8 接管):轮数上限=max_agent_steps(默认 6),
    react_max_iterations 键删除不留别名;token 预算 8000 不动。"""
    assert Settings.model_fields["max_agent_steps"].default == 6
    assert "react_max_iterations" not in Settings.model_fields
    assert Settings.model_fields["react_token_budget"].default == 8000


async def test_agent_never_binds_create_ticket():
    """终审 F 批:D2「Agent 自动建单方案作废」+ 用户钉「点『建工单』才写 tickets」
    → react 绑定集必须排除 create_ticket(建单唯一入口=前端按钮→POST /api/tickets)。"""
    model = ScriptStreamModel([_turn("答案")])
    [e async for e in react_agent_stream(state_with_evidence(), st(), model)]
    names = {t.name for t in model.bound_with}
    assert "create_ticket" not in names
    assert names == {"query_order", "query_product", "query_logistics", "query_faq"}


async def test_hallucinated_create_ticket_blocked_before_executor(monkeypatch):
    """执行闸:模型幻调 create_ticket 也绝不到执行器(即便按名可查),仅回 ok=False。"""
    fakes = _install_fake_tools(monkeypatch)
    fakes["create_ticket"] = FakeTool({"ticket_no": "T_SHOULD_NOT_EXIST"})
    model = ScriptStreamModel([
        _turn("我建个单。", [("create_ticket",
                             {"conversation_id": 42, "description": "x", "ticket_type": "投诉"},
                             "ct1")]),
        _turn("按钮在下方，请确认后点击。"),
    ])
    events = [e async for e in react_agent_stream(state_with_evidence(), st(), model)]
    assert fakes["create_ticket"].calls == 0, "create_ticket 真执行了(违 D2/需求8)"
    result = next(d for k, d in events if k == "tool_result")
    assert result["name"] == "create_ticket" and result["ok"] is False
