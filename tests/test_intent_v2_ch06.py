"""ch06 T2: 意图四件套——八类表/refund 出口/置信度解析(Review Focus 3)/
槽位正则/意图节点(快路/其他兜底/降级路小→大)。纯函数零依赖,节点面 ScriptModel。"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.workflows.routing import (
    INTENTS,
    extract_order_id,
    match_order_selection,
    parse_intent_json,
    route_for_intent,
)


def test_eight_intents_and_refund_route():
    assert "其他" in INTENTS and len(INTENTS) == 8
    assert route_for_intent("退款退货") == "refund"
    assert route_for_intent("售后") == "refund"
    assert route_for_intent("物流") == "data"
    assert route_for_intent("商品咨询") == "knowledge"


def test_other_and_unknown_land_on_knowledge():
    # 拍板 P1:「其他」与未知值都走带闸知识出口
    assert route_for_intent("其他") == "knowledge"
    assert route_for_intent("没这类") == "knowledge"
    assert route_for_intent("") == "knowledge"


def test_parse_requires_intent_and_confidence():
    assert parse_intent_json('{"intent": "物流", "confidence": 0.9}') == ("物流", 0.9)


def test_parse_accepts_numeric_string_confidence():
    assert parse_intent_json('{"intent": "物流", "confidence": "0.8"}') == ("物流", 0.8)


@pytest.mark.parametrize("raw", [
    '{"intent": "物流"}',                            # 缺 confidence
    '{"intent": "物流", "confidence": "高"}',          # 非数值
    '{"intent": "物流", "confidence": 1.5}',           # 越界
    '{"intent": "物流", "confidence": -0.1}',
    '{"intent": "转账", "confidence": 0.9}',           # 非法类别
    '{"category": "物流", "confidence": 0.9}',         # 缺 intent
    "我看这是物流问题 0.9",                             # 无 JSON
    "",
])
def test_parse_rejects_malformed(raw):
    assert parse_intent_json(raw) is None


def test_parse_tolerates_fence_and_prose():
    assert parse_intent_json('```json\n{"intent":"投诉","confidence":0.95}\n```') \
        == ("投诉", 0.95)
    assert parse_intent_json('我的判断是 {"intent": "售后", "confidence": 0.6} 。') \
        == ("售后", 0.6)


def test_slot_regexes():
    assert match_order_selection("我选择订单 1002") == "1002"
    assert match_order_selection("我选择订单1002") == "1002"
    assert match_order_selection("我选 1002") is None           # 非协议句
    assert match_order_selection("我选择订单 1002 谢谢") is None  # 必须全匹配
    assert extract_order_id("订单 1001 的物流", None) == "1001"
    assert extract_order_id("订单#1003：发货了吗") == "1003"
    assert extract_order_id("订单：1002", "") == "1002"
    assert extract_order_id("我想退款", "这个能退吗") is None


# ---- 节点面(make_intent_node(model, settings)):快路/兜底其他/降级路 ----

import logging  # noqa: E402

from app import workflows  # noqa: E402,F401  (确保包路径已加载,monkeypatch 用)
from app.workflows import nodes as N  # noqa: E402
from app.workflows.nodes import make_intent_node  # noqa: E402

pytestmark = pytest.mark.asyncio(loop_scope="function")


class ScriptModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def ainvoke(self, msgs):
        self.calls += 1
        self.last_msgs = list(msgs)
        if isinstance(r := self.replies.pop(0), Exception):
            raise r
        return AIMessage(content=r)


def _settings(**over):
    base = {"intent_small_model": "", "intent_confidence_threshold": 0.75}
    return SimpleNamespace(**{**base, **over})


def _state(q, resolved=None):
    st = {"messages": [HumanMessage(content=q)], "user_query": q}
    if resolved:
        st["resolved_query"] = resolved
    return st


OK_JSON = '{"intent": "物流", "confidence": 0.92}'


async def test_prompt_four_piece_contract():
    from app.prompts.intent import INTENT_PROMPT
    msgs = INTENT_PROMPT.format_messages(question="退货政策是什么")
    sys = msgs[0].content
    assert all(i in sys for i in INTENTS)           # 枚举选项(含「其他」)
    assert "confidence" in sys                      # 强制 JSON 两字段
    assert "退货政策是什么" in sys                    # 边界 few-shot 在场
    assert "0.92" not in sys                        # 示例不含测试题答案泄漏
    assert msgs[1].content == "退货政策是什么"


async def test_normal_json_sets_confidence_and_route():
    node = make_intent_node(ScriptModel([OK_JSON]), _settings())
    upd = await node(_state("订单1001到哪了", resolved="订单1001到哪了"))
    assert upd["intent"] == "物流" and upd["route"] == "data"
    assert upd["intent_confidence"] == 0.92
    assert upd["log"]["intent"] == "物流" and upd["log"]["confidence"] == 0.92


async def test_fast_path_uses_raw_user_query_zero_calls():
    m = ScriptModel([])
    upd = await make_intent_node(m, _settings())(
        _state("你好", resolved="你好"))
    assert upd["intent"] == "闲聊" and upd["route"] == "chitchat"
    assert m.calls == 0 and upd["log"]["fast_path"] is True


async def test_parse_fail_twice_falls_back_to_other():
    m = ScriptModel(["not json", "still not"])
    upd = await make_intent_node(m, _settings())(_state("随便说点什么"))
    assert upd["intent"] == "其他" and upd["route"] == "knowledge"
    assert upd["intent_confidence"] == 0.0
    assert m.calls == 2                              # 重试一次(ch05 P2 语义保留)


async def test_small_model_high_confidence_adopted(monkeypatch):
    small = ScriptModel([OK_JSON])
    big = ScriptModel(["不该被消费"])
    monkeypatch.setattr(N, "get_model", lambda st, *, model_name: small)
    upd = await make_intent_node(big, _settings(intent_small_model="tiny"))(
        _state("订单1001到哪了"))
    assert upd["intent"] == "物流" and small.calls == 1 and big.calls == 0


async def test_small_model_low_confidence_escalates(monkeypatch):
    small = ScriptModel(['{"intent": "物流", "confidence": 0.2}'])
    big = ScriptModel(['{"intent": "订单", "confidence": 0.99}'])
    monkeypatch.setattr(N, "get_model", lambda st, *, model_name: small)
    upd = await make_intent_node(big, _settings(intent_small_model="tiny"))(
        _state("订单1001到哪了"))
    assert upd["intent"] == "订单" and upd["intent_confidence"] == 0.99
    assert small.calls == 1 and big.calls == 1


async def test_small_model_error_escalates_directly(monkeypatch):
    small = ScriptModel([RuntimeError("429")])
    big = ScriptModel([OK_JSON])
    monkeypatch.setattr(N, "get_model", lambda st, *, model_name: small)
    upd = await make_intent_node(big, _settings(intent_small_model="tiny"))(
        _state("订单1001到哪了"))
    assert upd["intent"] == "物流" and big.calls == 1


async def test_logging_line_carries_resolved_and_confidence(caplog):
    st = {"user_query": "这个能退吗", "resolved_query": "订单1001的猫粮能退吗",
          "intent": "退款退货", "intent_confidence": 0.88,
          "log": {"nodes": ["coref", "intent"], "intent": "退款退货",
                  "confidence": 0.88, "gate_pass": True}}
    with caplog.at_level(logging.INFO, logger="app.workflows.nodes"):
        N.logging_node(st)
    line = caplog.records[-1].getMessage()
    assert "'nodes'" in line and "订单1001的猫粮能退吗" in line and "0.88" in line
