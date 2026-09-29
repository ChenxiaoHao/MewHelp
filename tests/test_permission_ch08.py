"""ch08 T4:权限闸(spec 权限闸节)。MCP 声明不可信:P5 恒只读由 T2 构造保证,
此处钉 write 无凭证拒/有凭证放行/拒绝不落审计(终局走确认流)。"""
from langchain_core.tools import tool

from app.tools.audit import AuditRecord
from app.tools.executor import ToolContext, execute_tool
from app.tools.registry import ToolSpec

REC = []


async def _sink(rec: AuditRecord):
    REC.append(rec)


@tool
async def fake_write(description: str) -> dict:
    """写。"""
    return {"ticket_no": "T1"}


def _wspec():
    return ToolSpec(fake_write, "write", "builtin")


async def test_write_without_credential_refused():
    out = await execute_tool(_wspec(), {"description": "x"}, "c1",
                             ToolContext(audit_sink=_sink))
    assert not out.ok and out.awaiting_confirmation
    assert "预览卡片确认" in out.result["error"]
    assert not REC                       # 等待确认=中间态,审计按终局(spec 审计节)


async def test_write_with_credential_executes():
    out = await execute_tool(_wspec(), {"description": "x"}, "c2",
                             ToolContext(ticket_confirmed=True, audit_sink=_sink))
    assert out.ok and out.result["ticket_no"] == "T1"
    assert REC[-1].status == "成功"      # T4 即接成功行(brief 测试钉死;T5 补失败/超时面)


async def test_invalid_write_hits_validation_not_permission():
    # 闸序=校验→权限:必填缺失先回「参数不合法」(验收4 追问路径的机制地基)
    out = await execute_tool(_wspec(), {}, "c3", ToolContext(audit_sink=_sink))
    assert "参数不合法" in out.result["error"] and not out.awaiting_confirmation
