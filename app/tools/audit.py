"""工具调用审计 sink(spec 审计节):AuditRecord 逐列对 DDL=db/init/08。

db_audit_sink 引擎未初始化(RuntimeError)/写败只 WARN 丢行——审计失败不许
反过来拦工具执行(需求5 红线);executor 层再兜一层 try/except 双保险。
"""

import logging
from dataclasses import asdict, dataclass

from app.db import crud
from app.db.engine import get_session_factory

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    conversation_id: int | None
    tool_call_id: str | None
    tool_name: str
    tool_source: str          # "builtin" | "mcp"
    mcp_server: str | None
    arguments: dict | None
    result_summary: str
    status: str               # 成功/失败/超时/校验拦下/权限拒绝
    error_message: str | None
    retry_count: int
    duration_ms: int | None


async def db_audit_sink(record: AuditRecord) -> None:
    try:
        factory = get_session_factory()
    except RuntimeError:
        logger.warning("audit dropped (engine unavailable) name=%s", record.tool_name)
        return
    try:
        async with factory() as session:
            await crud.insert_tool_audit(session, **asdict(record))
    except Exception:  # noqa: BLE001 —— 写败 WARN,上抛由 executor 再兜一层
        logger.warning("audit write failed name=%s", record.tool_name, exc_info=True)
