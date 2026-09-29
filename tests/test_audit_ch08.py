"""ch08 T1:审计行形状 + sink 降级语义(引擎未初始化不拦执行)。

DDL=db/init/08_ch08_tool_audit.sql(spec 附录二逐字);status 中文枚举列,
任何乱码即 charset 事故。集成测打活库(docker 3307,dev 常备)。
"""
from dataclasses import asdict

import pytest

from app.db import crud, models
from app.tools import audit


def _record(**kw):
    base = dict(conversation_id=1, tool_call_id="c1", tool_name="query_order",
                tool_source="builtin", mcp_server=None, arguments={"order_id": "1001"},
                result_summary="运输中", status="成功", error_message=None,
                retry_count=0, duration_ms=12)
    base.update(kw)
    return audit.AuditRecord(**base)


def test_model_columns_match_ddl():
    cols = {c.name: c for c in models.ToolAuditLog.__table__.columns}
    assert set(cols) == {"id", "conversation_id", "tool_call_id", "tool_name",
                         "tool_source", "mcp_server", "arguments", "result_summary",
                         "status", "error_message", "retry_count", "duration_ms",
                         "created_at"}
    assert not models.ToolAuditLog.__table__.foreign_keys  # 无 FK 红线(spec 审计节)
    assert cols["status"].type.enums == ["成功", "失败", "超时", "校验拦下", "权限拒绝"]


async def test_db_sink_engine_unavailable_warns(caplog, monkeypatch):
    # 引擎不可用:get_session_factory 抛 → WARN 丢行,绝不 raise
    # (需求5:审计失败不许反过来拦工具执行)
    # M1-F3:前提显式钉死(raiser 桩),否则同进程引擎已初始化时本测会向活库插脏行
    def boom():
        raise RuntimeError("engine not configured")
    monkeypatch.setattr(audit, "get_session_factory", boom)
    await audit.db_audit_sink(_record())
    assert any("audit" in r.getMessage().lower() for r in caplog.records)


@pytest.mark.integration
async def test_seam_insert_roundtrip_live_db():
    """活库逐列往返:中文 status 不乱码,JSON arguments 原样,自增 id 生效。"""
    from sqlalchemy import select

    from app.core.config import get_settings
    from app.db.engine import dispose_engine, get_session_factory, init_engine
    from app.db.models import ToolAuditLog
    init_engine(get_settings())          # ch07 集成同法:测试自举引擎
    try:  # M1-F1:不 dispose 会毒死同进程后续活库测(ch07 fixture 同法收口)
        async with get_session_factory()() as session:
            await crud.insert_tool_audit(session, **asdict(_record(
                tool_call_id="seam-t1", status="校验拦下")))
            row = (await session.execute(select(ToolAuditLog).where(
                ToolAuditLog.tool_call_id == "seam-t1"))).scalar_one()
            assert row.status == "校验拦下"
            assert row.arguments == {"order_id": "1001"}
            assert row.id > 0
            await session.delete(row)
            await session.commit()
    finally:
        await dispose_engine()
