"""ch06 T1: coref 正式版节点——首轮零调用透传 / 有历史走 LLM 补全 / 失败降级透传 /
每轮复位集合扩到 ch06 新键(pending_flow 先读后清)。"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.workflows.nodes import make_coref_node


class ScriptModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def ainvoke(self, msgs):
        self.calls += 1
        self.last_msgs = list(msgs)
        return AIMessage(content=self.replies.pop(0))


def _state(human_contents, **extra):
    msgs = [HumanMessage(content=h) for h in human_contents]
    return {"messages": msgs, "user_query": msgs[-1].content, **extra}


async def test_first_turn_passthrough_zero_calls():
    m = ScriptModel([])
    upd = await make_coref_node(m)(_state(["订单1001到哪了"]))
    assert upd["resolved_query"] == "订单1001到哪了"
    assert m.calls == 0 and upd["log"]["coref"] == "passthrough"


async def test_history_llm_completion():
    m = ScriptModel(["订单1001的猫粮能退吗"])
    st = _state(["我买了订单1001的猫粮", "到了", "这个能退吗"],
                evidence=[{"chunk_id": 1, "score": 0.9, "text": "旧证据"}])
    upd = await make_coref_node(m)(st)
    assert upd["resolved_query"] == "订单1001的猫粮能退吗"
    assert upd["log"]["coref"] == "done"
    assert upd["evidence"] == []            # ch05 复位语义照常
    assert upd["log"]["nodes"] == ["coref"]
    # 历史渲染进 prompt:补全类消息必须见到前轮实体
    rendered = "\n".join(getattr(x, "content", "") for x in m.last_msgs)
    assert "订单1001的猫粮" in rendered and "这个能退吗" in rendered


async def test_llm_failure_degrades_to_passthrough():
    class Boom:
        async def ainvoke(self, msgs):
            raise RuntimeError("net")
    upd = await make_coref_node(Boom())(_state(["上轮", "这个呢"]))
    assert upd["resolved_query"] == "这个呢"
    assert upd["log"]["coref"] == "degraded"


async def test_empty_model_output_falls_back_to_original():
    m = ScriptModel(["   "])
    upd = await make_coref_node(m)(_state(["上轮", "它能退吗"]))
    assert upd["resolved_query"] == "它能退吗"


async def test_reset_clears_ch06_keys_but_consumes_pending():
    m = ScriptModel(["它能退吗"])              # 非选择句:照常走 LLM,pending 读后即清
    st = _state(["上轮", "它能退吗"], pending_flow="refund",
                slot_order_id="1001", orders_payload=[{"x": 1}])
    upd = await make_coref_node(m)(st)
    assert upd["pending_flow"] == "" and upd["slot_order_id"] == ""
    assert upd["orders_payload"] == [] and upd["intent_confidence"] == 0.0
    assert upd["order_data"] == {} and upd["expanded_queries"] == []
