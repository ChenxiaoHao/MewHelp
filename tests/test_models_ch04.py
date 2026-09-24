"""ORM ↔ 05_ch04_schema.sql(=spec 附录 A)列级对账;活库对账走文末集成(ch03 同法)。"""

import pytest


def test_low_confidence_question_columns():
    from app.db.models import LowConfidenceQuestion

    t = LowConfidenceQuestion.__table__
    assert t.name == "low_confidence_questions"
    assert [c.name for c in t.columns] == [
        "id", "conversation_id", "raw_question", "source", "reason", "created_at",
    ]
    cols = t.columns
    assert cols["conversation_id"].nullable is True
    assert {fk.target_fullname for fk in cols["conversation_id"].foreign_keys} == {"conversations.id"}
    assert set(cols["source"].type.enums) == {"retrieval_low_conf", "self_check", "user_feedback"}
    assert cols["raw_question"].nullable is False and cols["source"].nullable is False


def test_faith_case_columns():
    from app.db.models import FaithCase

    t = FaithCase.__table__
    assert t.name == "faith_cases"
    assert [c.name for c in t.columns] == [
        "id", "eval_id", "bucket", "query", "strategy", "answer", "reason", "citations",
        "judge_model", "status", "seen_count", "first_seen_at", "last_seen_at",
        "resolution", "resolved_at",
    ]
    cols = t.columns
    assert cols["eval_id"].unique is True and cols["eval_id"].nullable is False
    assert cols["query"].type.length == 512 and cols["resolution"].type.length == 300
    assert set(cols["status"].type.enums) == {"未解决", "已解决", "无需解决"}
    assert cols["citations"].nullable is True and cols["judge_model"].nullable is True
    assert cols["seen_count"].nullable is False
    assert cols["resolved_at"].nullable is True


@pytest.mark.integration
async def test_live_schema_matches_ch04_orm():
    """需 mysql 容器 + 已执行 05(本任务 Step 8)。"""
    from sqlalchemy import inspect

    from app.core.config import get_settings
    from app.db.engine import dispose_engine, get_engine, init_engine
    from app.db.models import FaithCase, LowConfidenceQuestion

    init_engine(get_settings())
    try:
        def _reflect(sync_conn):
            insp = inspect(sync_conn)
            return {
                name: sorted(c["name"] for c in insp.get_columns(name))
                for name in (LowConfidenceQuestion.__tablename__, FaithCase.__tablename__)
            }

        async with get_engine().connect() as conn:
            live = await conn.run_sync(_reflect)
        for model in (LowConfidenceQuestion, FaithCase):
            cols = {c.name for c in model.__table__.columns}
            assert set(live[model.__tablename__]) == cols, model.__tablename__
    finally:
        await dispose_engine()
