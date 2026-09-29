"""ch08 T5:MCP 结果投影(挑字段/枚举翻话)+ 内置形状不动(spec 执行引擎节)。"""
from types import SimpleNamespace

from app.tools import executor as ex
from app.tools.executor import format_result
from app.tools.registry import ToolSpec


def _mcp_fake():
    return SimpleNamespace(name="query_waimai", description="", args={})


def _builtin_fake():
    return SimpleNamespace(name="query_order", description="", args={})


_RAW = {"carrier": "中通", "current_status": "TRANSPORT", "_internal": 1,
        "nodes": [{"n": i} for i in range(8)]}


def test_status_labels_defined_once_in_executor():
    # 测试只引用不复制字面(计划规4):键必须齐
    for k in ("COLLECTED", "TRANSPORT", "DELIVERING", "SIGNED",
              "APPROVED", "RECEIVING", "REFUNDING", "CLOSED"):
        assert k in ex.STATUS_LABELS
    assert ex.STATUS_LABELS["TRANSPORT"] == "运输中"


def test_format_projection_and_labels():
    spec = ToolSpec(_mcp_fake(), "readonly", "mcp", "logistics")
    out = format_result(spec, _RAW)
    assert out["current_status"] == ex.STATUS_LABELS["TRANSPORT"]
    assert "_internal" not in out
    assert len(out["nodes"]) == 5                      # 只留最近 5 条
    assert out is not _RAW                             # 投影出新 dict,不改原结果


def test_builtin_shape_untouched():
    builtin = ToolSpec(_builtin_fake(), "readonly", "builtin")
    assert format_result(builtin, _RAW) is _RAW        # 内置形状不动(现状兼容红线)


def test_non_dict_mcp_result_passthrough():
    spec = ToolSpec(_mcp_fake(), "readonly", "mcp", "logistics")
    assert format_result(spec, "纯文本") == "纯文本"    # Review Focus 3 面
