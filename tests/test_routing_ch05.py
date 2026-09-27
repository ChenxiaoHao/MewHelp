"""ch05 Task 2 → ch06 Task 2 retarget: 八意图→五出口分流纯函数 + 寒暄快路 + 意图 JSON 容错解析。

Review Focus 1 的钉死位:畸形 JSON(围栏/散文包裹/缺键/非法类别/非JSON)一律
返回 None 由调用方走兜底,绝不抛穿。ch06 收紧:缺 confidence/越界/布尔 同属畸形。
"""

import pytest

from app.prompts.intent import INTENT_PROMPT
from app.workflows.routing import (
    INTENTS,
    chitchat_fast_path,
    parse_intent_json,
    route_for_intent,
)


@pytest.mark.parametrize("intent,route", [
    ("商品咨询", "knowledge"), ("其他", "knowledge"),
    ("物流", "data"), ("订单", "data"),
    ("退款退货", "refund"), ("售后", "refund"),
    ("投诉", "complaint"), ("闲聊", "chitchat"),
])
def test_eight_intents_map_to_five_routes(intent, route):
    assert route_for_intent(intent) == route


def test_unknown_intent_falls_back_to_knowledge():
    # 拍板 P1/P2:未知与「其他」归知识类(强制检索+置信度闸,防幻觉代价最小)
    assert route_for_intent("转账") == "knowledge"
    assert route_for_intent("") == "knowledge"


def test_intents_tuple_has_eight_named_categories():
    assert INTENTS == ("物流", "订单", "商品咨询", "退款退货", "售后",
                       "投诉", "闲聊", "其他")


def test_parse_intent_json_plain():
    assert parse_intent_json('{"intent": "物流", "confidence": 0.9}') == ("物流", 0.9)


def test_parse_intent_json_tolerates_code_fence():
    raw = '```json\n{"intent": "物流", "confidence": 0.8}\n```'
    assert parse_intent_json(raw) == ("物流", 0.8)


def test_parse_intent_json_tolerates_surrounding_prose():
    raw = '好的,我的判断是 {"intent": "投诉", "confidence": 0.6} 。'
    assert parse_intent_json(raw) == ("投诉", 0.6)


def test_parse_intent_json_accepts_string_number_confidence():
    # 字符串数字放过(小模型常见输出形态),越界仍拒
    assert parse_intent_json('{"intent": "订单", "confidence": "0.42"}') == ("订单", 0.42)
    assert parse_intent_json('{"intent": "订单", "confidence": "42"}') is None


@pytest.mark.parametrize("raw", [
    '{"intent": "转账", "confidence": 0.9}',   # 非法类别
    '{"intent": 物流, "confidence": 0.9}',     # 非合法 JSON
    '{"category": "物流", "confidence": 0.9}', # 缺 intent 键
    '{"intent": "物流"}',                       # 缺 confidence(ch06 收紧)
    '{"intent": "物流", "confidence": 1.5}',    # 越界
    '{"intent": "物流", "confidence": -0.1}',   # 越界
    '{"intent": "物流", "confidence": true}',   # 布尔不是置信度
    '{"intent": "物流", "confidence": "很高"}', # 非数值
    "我看这是物流问题",                           # 无 JSON
    "",                                        # 空串
])
def test_parse_intent_json_rejects_malformed(raw):
    assert parse_intent_json(raw) is None


@pytest.mark.parametrize("text", [
    "你好", "你好！", " 您好 ", "hello", "HI", "在吗", "哈喽", "早上好~", "谢谢",
])
def test_fast_path_hits_pure_greetings(text):
    assert chitchat_fast_path(text) is True


@pytest.mark.parametrize("text", [
    "你好，订单1001到哪了", "退款", "你好吗，我买的东西坏了要售后", "",
    "你好你好你好你好你好你好你好你好你好你好",  # 超长度护栏不误吞
])
def test_fast_path_misses_real_business(text):
    assert chitchat_fast_path(text) is False


def test_intent_prompt_lists_all_eight_intents():
    rendered = INTENT_PROMPT.messages[0].prompt.template
    assert all(i in rendered for i in INTENTS)
    assert '{"intent"' in rendered  # 输出契约在 system 里钉死
    assert "confidence" in rendered  # ch06 四件套:恰好两字段契约
    assert "别硬塞业务意图" in rendered  # 「其他」兜底口径在场


def test_intent_prompt_renders_without_keyerror():
    """回归钉死:system 里的 JSON 示例花括号必须 f-string 转义(双花括号)。
    单花括号写法在 ChatPromptTemplate.format 时抛 KeyError——样例评估实测翻车点。"""
    msgs = INTENT_PROMPT.format_messages(question="你好呀")
    assert '{"intent"' in msgs[0].content
    assert "你好呀" in msgs[1].content
