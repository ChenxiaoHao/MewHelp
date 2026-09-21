"""ORM ↔ spec 附录 A 结构一致性(纯元数据断言,不连库;列级对账另见文末集成测试)。"""

import pytest


def test_knowledge_chunk_columns():
    from app.db.models import KnowledgeChunk

    cols = KnowledgeChunk.__table__.columns
    assert [c.name for c in cols] == [
        "id", "category", "questions", "answer", "section_path", "content_type",
        "is_key_clause", "prev_chunk_id", "next_chunk_id", "vector_id",
        "vectorize_status", "created_at", "updated_at",
    ]
    t = KnowledgeChunk.__table__
    assert t.name == "knowledge_chunks"
    assert cols["category"].nullable is False and cols["category"].type.length == 255
    assert cols["questions"].nullable is False
    assert cols["section_path"].nullable is True and cols["section_path"].type.length == 512
    assert cols["vector_id"].nullable is True and cols["vector_id"].type.length == 64
    assert cols["is_key_clause"].nullable is False
    assert cols["vectorize_status"].nullable is False
    # 自引用外键两枚(DDL 侧 ON DELETE SET NULL 在集成对账脚本核)
    fks = {fk.target_fullname for fk in cols["prev_chunk_id"].foreign_keys}
    assert fks == {"knowledge_chunks.id"}  # 计划原文 tuple(split("."))[0] 会截成表名,属测试笔误,已取全名比对
    assert cols["next_chunk_id"].foreign_keys


def test_staging_columns():
    from app.db.models import QaExtractionStaging

    t = QaExtractionStaging.__table__
    assert t.name == "qa_extraction_staging"
    names = [c.name for c in t.columns]
    assert names == ["id", "batch_no", "source_ref", "question", "answer", "status", "created_at"]
    assert t.columns["source_ref"].nullable is True
    assert t.columns["status"].nullable is False


@pytest.mark.integration
async def test_live_schema_matches_orm():
    """真库列名与 ORM 对账(需 mysql 容器 + 已执行 03/04)。用 get_settings 而非 fake_settings:集成测试打真实 .env 库。"""
    from sqlalchemy import inspect

    from app.core.config import get_settings
    from app.db.engine import dispose_engine, get_engine, init_engine
    from app.db.models import KnowledgeChunk, QaExtractionStaging

    init_engine(get_settings())
    try:
        def _reflect(sync_conn):
            # Inspector 是惰性的:get_columns 的 IO 必须发生在 run_sync 回调内部,
            # 回调外再调用会 MissingGreenlet(执行期实测,计划自审版踩过)
            insp = inspect(sync_conn)
            return {
                name: sorted(c["name"] for c in insp.get_columns(name))
                for name in (KnowledgeChunk.__tablename__, QaExtractionStaging.__tablename__)
            }

        async with get_engine().connect() as conn:
            live = await conn.run_sync(_reflect)
        for model in (KnowledgeChunk, QaExtractionStaging):
            cols = {c.name for c in model.__table__.columns}
            assert set(live[model.__tablename__]) == cols, model.__tablename__
    finally:
        await dispose_engine()
