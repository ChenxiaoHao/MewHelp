"""ch08 T3:JSON Schema 校验闸(spec 校验闸节)——拦下不抛异常,错误回灌+审计。"""
from langchain_core.tools import tool

from app.tools.audit import AuditRecord
from app.tools.executor import ToolContext, audit_denied, execute_tool
from app.tools.registry import ToolSpec

REC = []


async def _sink(rec: AuditRecord):
    REC.append(rec)


@tool
async def probe(order_id: str, qty: int = 1) -> dict:
    """查一件东西。order_id: 订单号。"""
    return {"ok": True, "order_id": order_id, "qty": qty}


def _spec():
    return ToolSpec(probe, "readonly", "builtin")


def _ticket_probe():
    from app.tools.definitions import create_ticket
    return create_ticket


async def test_missing_required_blocks_with_chinese_reason():
    out = await execute_tool(_spec(), {"qty": 2}, "c1", ToolContext(audit_sink=_sink))
    assert not out.ok and "参数不合法" in out.result["error"]
    assert "order_id" in out.result["error"]           # 指名缺的字段
    assert REC and REC[-1].status == "校验拦下"


async def test_wrong_type_blocked():
    out = await execute_tool(_spec(), {"order_id": 123}, "c2",
                             ToolContext(audit_sink=_sink))
    assert not out.ok and "参数不合法" in out.result["error"]
    assert REC[-1].status == "校验拦下"


async def test_valid_args_pass_through():
    out = await execute_tool(_spec(), {"order_id": "1001"}, "c3",
                             ToolContext(audit_sink=_sink))
    assert out.ok and out.result["order_id"] == "1001"


async def test_literal_enum_out_of_range_blocked():
    out = await execute_tool(
        ToolSpec(_ticket_probe(), "readonly", "builtin"),
        {"description": "x", "ticket_type": "爆炸"}, "c4", ToolContext(audit_sink=_sink))
    assert not out.ok and REC[-1].status == "校验拦下"


async def test_audit_denied_shape():
    """Review Focus 6 生产面:幻觉未登记调用按 builtin 命名空间记权限拒绝。"""
    await audit_denied(ToolContext(conversation_id=9, audit_sink=_sink),
                       "query_weather", "c9", {"city": "hz"})
    rec = REC[-1]
    assert rec.status == "权限拒绝" and rec.tool_source == "builtin"
    assert rec.tool_name == "query_weather" and rec.mcp_server is None
    assert rec.conversation_id == 9 and rec.error_message
