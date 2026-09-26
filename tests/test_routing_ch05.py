"""ch05 Task 2: 七意图→四出口分流纯函数 + 寒暄快路 + 意图 JSON 容错解析。

Review Focus 1 的钉死位:畸形 JSON(围栏/散文包裹/缺键/非法类别/非JSON)一律
返回 None 由调用方走兜底,绝不抛穿。
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
    ("商品咨询", "knowledge"), ("退款退货", "knowledge"),
    ("物流", "data"), ("订单", "data"), ("售后", "data"),
    ("投诉", "complaint"), ("闲聊", "chitchat"),
])
def test_seven_intents_map_to_four_routes(intent, route):
    assert route_for_intent(intent) == route


def test_unknown_intent_falls_back_to_knowledge():
    # 拍板 P2:兜底归知识类(强制检索+置信度闸,防幻觉代价最小)
    assert route_for_intent("转账") == "knowledge"
    assert route_for_intent("") == "knowledge"


def test_intents_tuple_has_seven_named_categories():
    assert INTENTS == ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")


def test_parse_intent_json_plain():
    assert parse_intent_json('{"intent": "物流"}') == "物流"


def test_parse_intent_json_tolerates_code_fence():
    assert parse_intent_json('```json\n{"intent": "物流"}\n```') == "物流"


def test_parse_intent_json_tolerates_surrounding_prose():
    assert parse_intent_json('好的,我的判断是 {"intent": "投诉"} 。') == "投诉"


@pytest.mark.parametrize("raw", [
    '{"intent": "转账"}',      # 非法类别
    '{"intent": 物流}',        # 非合法 JSON
    '{"category": "物流"}',    # 缺 intent 键
    "我看这是物流问题",         # 无 JSON
    "",                        # 空串
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


def test_intent_prompt_lists_all_seven_intents():
    rendered = INTENT_PROMPT.messages[0].prompt.template
    assert all(i in rendered for i in INTENTS)
    assert '{"intent"' in rendered  # 输出契约在 system 里钉死


def test_intent_prompt_renders_without_keyerror():
    """回归钉死:system 里的 JSON 示例花括号必须 f-string 转义(双花括号)。
    单花括号写法在 ChatPromptTemplate.format 时抛 KeyError——样例评估实测翻车点。"""
    msgs = INTENT_PROMPT.format_messages(question="你好呀")
    assert '{"intent"' in msgs[0].content
    assert "你好呀" in msgs[1].content
