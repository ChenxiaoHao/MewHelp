"""ch07 T3:层2 半压三规则 + 降级切割对齐 + ContextStore 读侧(spec「三层结构」表)。

Review Focus 1 钉测面:半压渲染**绝不产出 ToolMessage、绝不产出带 tool_calls 的
AIMessage**——工具链整体折叠成一行标识的 AIMessage 是唯一合法形态,否则上游
(OpenAI 兼容端)直接 400。切割对齐 human 轮边界=一轮不拆成半压+原文两态。
"""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.context.budget import estimate_items, estimate_text
from app.context.layers import pick_degrade_cut, render_layer2

SETTINGS = SimpleNamespace(assistant_head_chars=60)


def _row(rid, role, content=None, tool_calls=None):
    return SimpleNamespace(id=rid, role=role, content=content, tool_calls=tool_calls)


def test_user_line_verbatim():
    rows = [_row(1, "user", "我要退款，订单号 1001")]
    out = render_layer2(rows, SETTINGS)
    assert len(out) == 1 and isinstance(out[0], HumanMessage)
    assert out[0].content == "我要退款，订单号 1001"  # user 原话一个字不动


def test_assistant_head_kept_rest_dropped():
    body = "退" * 100
    out = render_layer2([_row(2, "assistant", body)], SETTINGS)
    assert isinstance(out[0], AIMessage)
    assert out[0].content == "退" * 60 + "…"


def test_short_assistant_no_ellipsis():
    out = render_layer2([_row(2, "assistant", "好的")], SETTINGS)
    assert out[0].content == "好的"


def test_tool_chain_folds_into_one_marker():
    result = "库存数据" * 100
    rows = [
        _row(3, "assistant", None, [{"id": "c1", "name": "query_stock", "args": {}}]),
        _row(4, "tool", result, None),
    ]
    rows[1].tool_call_id = "c1"
    out = render_layer2(rows, SETTINGS)
    assert len(out) == 1 and isinstance(out[0], AIMessage)
    assert out[0].content == f"[工具结果·query_stock·≈{estimate_text(result)} token 已折叠]"


def test_folded_assistant_with_text_keeps_head_then_marker():
    rows = [
        _row(3, "assistant", "帮您查一下" + "哈" * 80,
             [{"id": "c1", "name": "query_order", "args": {}}]),
        _row(4, "tool", "订单1001已发货" * 30, None),
    ]
    rows[1].tool_call_id = "c1"
    out = render_layer2(rows, SETTINGS)
    assert len(out) == 1
    text, marker = out[0].content.split("\n")
    assert text == "帮您查一下" + "哈" * 55 + "…"  # 头部合计 60 字
    assert marker.startswith("[工具结果·query_order·≈") and marker.endswith(" token 已折叠]")


def test_multi_tool_chain_all_folded():
    rows = [
        _row(3, "assistant", None, [
            {"id": "c1", "name": "query_faq", "args": {}},
            {"id": "c2", "name": "query_order", "args": {}}]),
        _row(4, "tool", "faq结果" * 50, None),
        _row(5, "tool", "order结果" * 50, None),
    ]
    rows[1].tool_call_id, rows[2].tool_call_id = "c1", "c2"
    out = render_layer2(rows, SETTINGS)
    assert len(out) == 1
    markers = out[0].content.split("\n")
    assert [m.split("·")[1] for m in markers] == ["query_faq", "query_order"]


def test_dangling_tool_call_chain_folds_to_nonempty_marker():
    """M1 评审 Minor→Important(on_tool_result 落库失败吞异常→tool 行缺失):
    悬空 tool_calls 链不得折成空 content AIMessage(严格兼容端拒空 assistant=400 类)。"""
    rows = [_row(3, "assistant", None, [{"id": "c1", "name": "query_order", "args": {}}])]
    out = render_layer2(rows, SETTINGS)
    assert len(out) == 1 and out[0].content != ""
    assert out[0].content == "[工具调用·query_order·已折叠]"


def test_render_never_produces_tool_or_toolcall_messages():
    rows = [
        _row(1, "user", "查单"),
        _row(2, "assistant", None, [{"id": "c1", "name": "query_order", "args": {}}]),
        _row(3, "tool", "结果", None),
        _row(4, "assistant", "已查到"),
    ]
    rows[2].tool_call_id = "c1"
    out = render_layer2(rows, SETTINGS)
    assert not any(isinstance(m, ToolMessage) for m in out)
    assert not any(getattr(m, "tool_calls", None) for m in out)


def test_orphan_tool_row_folds_not_dropped():
    """防御面:理论不存在的孤儿 tool 行也折成标识,绝不还原 ToolMessage。"""
    out = render_layer2([_row(9, "tool", "孤儿结果", None)], SETTINGS)
    assert len(out) == 1 and isinstance(out[0], AIMessage)
    assert not any(isinstance(m, ToolMessage) for m in out)


def _rows_turn(count_turns=3):
    """每轮 user+assistant+tool 三行,id 从 1 起;est 每行折 10。"""
    rows, rid = [], 1
    for _ in range(count_turns):
        rows.append(_row(rid, "user", "问")); rid += 1
        rows.append(_row(rid, "assistant", None, [{"id": f"c{rid}", "name": "t", "args": {}}])); rid += 1
        rows.append(_row(rid, "tool", "答", None)); rid += 1
    return rows


def test_cut_none_when_within_budget():
    assert pick_degrade_cut(_rows_turn(3), 999, estimate_items) is None


def test_cut_lands_before_retained_human_turn():
    rows = _rows_turn(3)  # 9 行,每行 1 字内容+4 开销=5,全批 45
    # 预算 22:从新往老留 4 行(20)不超、5 行(25)超 → 切割点须落在轮首 user 之前
    cut = pick_degrade_cut(rows, 22, estimate_items)
    assert cut is not None
    kept = [r for r in rows if r.id > cut]
    assert 0 < len(kept) < len(rows)
    assert kept[0].role == "user"            # 保留段首条必 human(一轮不拆两态)


def test_cut_aligns_forward_to_human_boundary_midturn():
    rows = _rows_turn(3)  # 每轮 est=user5+assistant4+tool5=14,后缀和 idx4 起=23>22
    # 纯预算起点 idx4 是轮中 assistant → 向新对齐到第3轮首 idx6 user;
    # 降级批=前两轮完整(轮不拆两态),保留段 14 ≤ 22(T4 实测:向老对齐=死循环 None)
    cut = pick_degrade_cut(rows, 22, estimate_items)
    assert cut == rows[5].id  # 新 layer1_from=第2轮 tool 行
    kept = [r for r in rows if r.id > cut]
    assert kept[0].role == "user" and estimate_items(kept) <= 22


def test_single_turn_over_budget_degrades_whole_batch():
    rows = _rows_turn(1)  # 预算连一整轮(14)都装不下:整批进层2 等摘要,层1 不留超限原文
    assert pick_degrade_cut(rows, 4, estimate_items) == rows[-1].id


def test_cut_degrades_whole_batch_when_no_forward_turn_start():
    """M1 评审#2:尾行是残缺轮(超预算单行 tool)且向新无轮首 → 整批降级而非 None,
    否则超限原文永远发给模型;交层2 摘要压批才是出路。"""
    rows = [_row(i + 1, "user" if i % 2 == 0 else "assistant", "x" * 50)
            for i in range(9)]
    rows.append(_row(10, "tool", "y" * 2000, None))  # 单行超预算(501+ 估算)
    assert rows[9].id == 10
    cut = pick_degrade_cut(rows, 500, estimate_items)
    assert cut == 10  # 全批进层2,层1 留空等当轮新句
