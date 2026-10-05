"""ch09 T3 活库逐列往返:09/10 应用后的数据面证据(integration)。

纸面半(seam 测)只证 DDL ⊇ ORM;这里证活 MySQL 真库:
①新表中文 ENUM/server_default 逐值往返不乱码;②LCQ 新列 JSON 原样回读;
③旧写方(add_low_confidence_question 原签名)ALTER 后仍可写,新列落 NULL;
④matched_review_id FK ON DELETE SET NULL 语义;⑤messages 快照列随行往返。
每例自造会话/行,随机 uid,测毕自清(ch07 活库集成同法)。
"""

import uuid

import pytest
from sqlalchemy import delete, select

from app.core.config import get_settings
from app.db import crud
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.db.models import (
    Conversation,
    EvalRuns,
    LowConfidenceQuestion,
    Message,
    ReviewQueue,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def session():
    init_engine(get_settings())
    try:
        async with get_session_factory()() as s:
            yield s
    finally:
        await dispose_engine()


def _uid() -> str:
    return f"it-ch09-{uuid.uuid4().hex[:10]}"


async def _mk_conv(session, uid: str) -> int:
    conv = Conversation(user_id=uid)
    session.add(conv)
    await session.commit()
    return conv.id


async def _purge(session, uid: str, conv_ids, review_ids, eval_ids) -> None:
    if conv_ids:
        lcq_ids = (await session.execute(
            select(LowConfidenceQuestion.id).where(
                LowConfidenceQuestion.conversation_id.in_(conv_ids)))).scalars().all()
        if lcq_ids:
            await session.execute(delete(LowConfidenceQuestion).where(
                LowConfidenceQuestion.id.in_(lcq_ids)))
        await session.execute(delete(Message).where(
            Message.conversation_id.in_(conv_ids)))
    if review_ids:
        await session.execute(delete(ReviewQueue).where(
            ReviewQueue.id.in_(review_ids)))
    if conv_ids:
        await session.execute(delete(Conversation).where(
            Conversation.id.in_(conv_ids)))
    if eval_ids:
        await session.execute(delete(EvalRuns).where(EvalRuns.id.in_(eval_ids)))
    await session.commit()


async def test_lcq_old_writer_and_new_columns_roundtrip(session):
    """旧写方原签名仍可写(Review Focus 面:ALTER 不伤 ch04/05/06 漏斗);
    新列直写 JSON 快照含中文,回读逐字节一致。"""
    uid = _uid()
    conv_id = await _mk_conv(session, uid)
    try:
        await crud.add_low_confidence_question(
            session, conversation_id=conv_id,
            raw_question="保修期内电池鼓包怎么换新", source="retrieval_low_conf",
            reason="top1=0.11")
        row = (await session.execute(
            select(LowConfidenceQuestion).where(
                LowConfidenceQuestion.conversation_id == conv_id))).scalar_one()
        assert row.retrieved_chunks is None and row.matched_review_id is None

        snap = [{"chunk_id": 123, "score": 0.42,
                 "text": "电池鼓包属于质保范围,请携带购机发票到服务中心"}]
        row.retrieved_chunks = snap
        await session.commit()
        await session.refresh(row)
        assert row.retrieved_chunks == snap, "JSON 快照往返漂移或中文乱码"
    finally:
        await _purge(session, uid, [conv_id], [], [])


async def test_lcq_matched_review_fk_set_null(session):
    review_ids: list[int] = []
    conv_id = None
    uid = _uid()
    try:
        conv_id = await _mk_conv(session, uid)
        rq = ReviewQueue(normalized_question="电池鼓包换新",
                         ai_suggested_answer="携发票到服务中心换新")
        session.add(rq)
        await session.commit()
        review_ids.append(rq.id)

        lcq = LowConfidenceQuestion(conversation_id=conv_id,
                                    raw_question="电池鼓包怎么换新",
                                    source="user_feedback", reason="seq=1",
                                    matched_review_id=rq.id)
        session.add(lcq)
        await session.commit()

        await session.delete(rq)
        await session.commit()
        await session.refresh(lcq)
        assert lcq.matched_review_id is None, "FK 未带 ON DELETE SET NULL(09_ DDL 面)"
    finally:
        ids = [conv_id] if conv_id else []
        if ids:
            await session.execute(delete(LowConfidenceQuestion).where(
                LowConfidenceQuestion.conversation_id.in_(ids)))
        if review_ids:
            await session.execute(delete(ReviewQueue).where(
                ReviewQueue.id.in_(review_ids)))
        await _purge(session, uid, ids, [], [])


async def test_review_queue_defaults_and_enum_roundtrip(session):
    """中文 ENUM 三值 + server_default(待审/1)在活库逐字往返。"""
    ids: list[int] = []
    rq = ReviewQueue(normalized_question="快递丢件赔付标准",
                     ai_suggested_answer="按声明价值赔付")
    session.add(rq)
    await session.commit()
    try:
        ids.append(rq.id)
        await session.refresh(rq)  # server_default 列不回客户端,须显式 reload
        assert rq.review_status == "待审" and rq.occurrence_count == 1, \
            "server_default 未在活库生效(09_ DDL 面)"
        rq.review_status = "驳回"
        rq.approved_answer = None
        await session.commit()
        await session.refresh(rq)
        assert rq.review_status == "驳回"
    finally:
        await session.execute(delete(ReviewQueue).where(ReviewQueue.id.in_(ids)))
        await session.commit()


async def test_eval_runs_metrics_json_roundtrip(session):
    """triggered_by 中文 ENUM + metrics JSON 原样回读(浮点精度不失)。"""
    ids: list[int] = []
    run = EvalRuns(triggered_by="手动", dataset_size=300,
                   metrics={"recall_at_3": 0.8125, "recall_at_10": 0.9375,
                            "mrr_at_10": 0.7419, "faithfulness": 0.96})
    session.add(run)
    await session.commit()
    try:
        ids.append(run.id)
        got = (await session.execute(
            select(EvalRuns).where(EvalRuns.id == run.id))).scalar_one()
        assert got.triggered_by == "手动"
        assert got.metrics == {"recall_at_3": 0.8125, "recall_at_10": 0.9375,
                               "mrr_at_10": 0.7419, "faithfulness": 0.96}
        assert got.created_at is not None
    finally:
        await session.execute(delete(EvalRuns).where(EvalRuns.id.in_(ids)))
        await session.commit()


async def test_messages_snapshot_roundtrip_and_legacy_writer(session):
    """旧形 Message(...) 不传快照仍可写(NULL);带快照行逐列回读一致。"""
    uid = _uid()
    conv_id = await _mk_conv(session, uid)
    try:
        session.add(Message(conversation_id=conv_id, role="user", content="电池鼓包"))
        snap = [{"chunk_id": 9, "score": 0.87, "text": "鼓包电池立即停用并联系售后"}]
        session.add(Message(conversation_id=conv_id, role="assistant",
                            content="请携发票换新", retrieval_snapshot=snap))
        await session.commit()
        rows = (await session.execute(
            select(Message).where(Message.conversation_id == conv_id)
            .order_by(Message.id))).scalars().all()
        assert len(rows) == 2
        assert rows[0].retrieval_snapshot is None, "旧写方缺列须落 NULL"
        assert rows[1].retrieval_snapshot == snap
    finally:
        await _purge(session, uid, [conv_id], [], [])
