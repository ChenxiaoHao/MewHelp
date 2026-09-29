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


async def test_malformed_external_schema_degrades_to_pass(caplog):
    """M2-I1:校验闸对外部 MCP 自带畸形 JSON Schema(非法 type 字面量)不逃逸——
    视作无可校验形状放行+WARN,让工具自身入参兜底(不可信外部声明面)。"""
    import logging
    caplog.set_level(logging.WARNING)

    class Bad:
        name = "bad"; description = ""
        # args 进 fallback 拼装(tool_call_schema 非 type 类时被忽略);非法 type
        # 字面量放进实例触达的属性上——jsonschema 惰性绑定,iter 即 UnknownType
        args = {"whatever": {"type": "not-a-type"}}

        async def ainvoke(self, args, config=None, **kw):
            return {"ok": True}

    out = await execute_tool(ToolSpec(Bad(), "readonly", "mcp", "srv"),
                             {"whatever": 1}, "v-b1", ToolContext(audit_sink=_sink))
    assert out.ok, out.result
    assert any("schema" in r.getMessage().lower() for r in caplog.records)


async def test_no_cross_call_schema_state():
    """M2-M1 升 I:_SCHEMA_CACHE 按 id() 键在 P4 每轮现拿形制下必炸——MCP 件
    轮末 GC、CPython 地址复用挂错 schema。校验闸必须无跨调用状态(删缓存)。"""
    from pydantic import BaseModel

    class M1(BaseModel):
        a: str

    class M2(BaseModel):
        b: str

    class Mut:
        name = "mut"; description = ""; args = {}

        async def ainvoke(self, args, config=None, **kw):
            return {"ok": True}

    t = Mut()
    t.tool_call_schema = M1   # isinstance(sc, type) 才走 model_json_schema 面
    spec = ToolSpec(t, "readonly", "builtin")
    out1 = await execute_tool(spec, {"a": "x"}, "v-m1", ToolContext(audit_sink=_sink))
    assert out1.ok
    t.tool_call_schema = M2
    out2 = await execute_tool(spec, {"b": "y"}, "v-m2", ToolContext(audit_sink=_sink))
    assert out2.ok, "schema 变更后校验须用新 schema(旧 id 缓存=错杀)"
