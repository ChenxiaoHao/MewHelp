"""ch06 Task 3: 扩写解析/证据合并纯函数 + 订单同播种数据面(P4 一致性钉)。

merge_evidence 的语义拍板 P3:chunk_id 去重取最高分、降序、截断 cap。
list_user_orders 与 query_order 共用 _make_order 同播种生成——卡片与详情逐字一致。
"""

from app.workflows.routing import merge_evidence, parse_queries_json


def test_parse_queries():
    assert parse_queries_json('{"queries": ["a", "b", "a", "c", "d", "e"]}') == ["a", "b", "c", "d"]
    assert parse_queries_json('```json\n{"queries":["x"]}\n```') == ["x"]
    assert parse_queries_json('{"queries": "x"}') is None
    assert parse_queries_json('{"q": ["x"]}') is None
    assert parse_queries_json('不是json') is None


def test_parse_queries_rejects_dirty_members():
    assert parse_queries_json('{"queries": ["x", 1]}') is None      # 非字符串成员
    assert parse_queries_json('{"queries": []}') is None            # 空数组无检索价值
    assert parse_queries_json('{"queries": ["  ", "x"]}') is None   # 空白成员


def test_merge_evidence_dedup_and_cap():
    g1 = [{"chunk_id": 1, "score": 0.5, "text": "t1"}, {"chunk_id": 2, "score": 0.9, "text": "t2"}]
    g2 = [{"chunk_id": 2, "score": 0.7, "text": "t2"}, {"chunk_id": 3, "score": 0.6, "text": "t3"}]
    out = merge_evidence([g1, g2], cap=2)
    assert [(c["chunk_id"], c["score"]) for c in out] == [(2, 0.9), (3, 0.6)]


def test_merge_evidence_empty_groups():
    assert merge_evidence([], cap=10) == []


async def test_orders_single_source_consistency():
    from app.tools.definitions import _make_order, list_user_orders
    cards = list_user_orders("demo_user")
    assert [c["order_id"] for c in cards] == ["1001", "1002", "1003"]
    for c in cards:
        full = _make_order(c["order_id"])
        assert (c["status"], c["amount"]) == (full["status"], full["amount"])  # P4 逐字一致
        assert len(c["items"]) == len(full["items"])


def test_expand_prompt_renders_without_keyerror():
    """回归钉死:system 里 JSON 示例花括号必须 f-string 转义(ch04 T4 同款坑)。"""
    from app.prompts.query_expand import EXPAND_PROMPT
    msgs = EXPAND_PROMPT.format_messages(question="我买的猫粮想退")
    assert '{"queries": [' in msgs[0].content
    assert "我买的猫粮想退" in msgs[1].content
