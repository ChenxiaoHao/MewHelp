"""ch10 T1:topic_classifications ORM——uk_question_id 一人一行幂等钉(integration 活库)。

DDL=db/init/11_ 用户原文逐字;fk→low_confidence_questions.id。
真池任一行 insert 归类成功;同 question_id 二次 insert 必 IntegrityError;
测后删行(真池本身只读不动)。
"""

import pytest
from sqlalchemy import delete, exc, select, text

from app.core.config import get_settings
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.db.models import TopicClassification


@pytest.fixture
async def session():
    init_engine(get_settings())
    try:
        async with get_session_factory()() as s:
            yield s
    finally:
        await dispose_engine()


@pytest.mark.integration
async def test_one_row_per_question_uk_enforced(session):
    qid = (await session.execute(
        text("SELECT id FROM low_confidence_questions ORDER BY id LIMIT 1"))).scalar()
    assert qid is not None, "活池至少一行(现状 138)"
    session.add(TopicClassification(question_id=qid, labels=["物流", "运费"]))
    await session.commit()

    session.add(TopicClassification(question_id=qid, labels=["发票"]))
    with pytest.raises(exc.IntegrityError):
        await session.commit()
    await session.rollback()

    rows = (await session.execute(
        select(TopicClassification).where(TopicClassification.question_id == qid)
    )).scalars().all()
    assert len(rows) == 1 and rows[0].labels == ["物流", "运费"]
    assert rows[0].classified_at is not None

    await session.execute(delete(TopicClassification)
                          .where(TopicClassification.question_id == qid))
    await session.commit()
