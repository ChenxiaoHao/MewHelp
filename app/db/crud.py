"""数据访问层：会话/消息/FAQ 检索/工单。运行期落库失败的降级策略在调用方（persister）。"""

import logging
from datetime import date, datetime

from sqlalchemy import Select, delete, func, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Conversation,
    FaithCase,
    Faq,
    KnowledgeChunk,
    LowConfidenceQuestion,
    Message,
    QaExtractionStaging,
    Ticket,
)

logger = logging.getLogger(__name__)


async def create_or_get_conversation(
    session: AsyncSession, conversation_id: int | None, user_id: str
) -> Conversation:
    """conversation_id 由服务端拥有：None 或查无 → 新建；查到 → 复用。"""
    if conversation_id is not None:
        conv = await session.get(Conversation, conversation_id)
        if conv is not None:
            return conv
    conv = Conversation(user_id=user_id)
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv


async def add_message(
    session: AsyncSession,
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )
    session.add(msg)
    await session.commit()
    return msg


def build_faq_query(keyword: str, limit: int = 3) -> Select:
    like = f"%{keyword}%"
    return (
        select(Faq)
        .where(or_(Faq.question.like(like), Faq.answer.like(like)))
        .limit(limit)
    )


async def search_faq(
    session: AsyncSession, keyword: str, limit: int = 3
) -> list[Faq]:
    rows = (await session.execute(build_faq_query(keyword, limit))).scalars().all()
    return list(rows)


def next_ticket_no(today_count: int, today: date) -> str:
    """T{YYYYMMDD}{当日已有工单数+1:03d}，如 T20260920001。"""
    return f"T{today:%Y%m%d}{today_count + 1:03d}"


async def create_ticket(
    session: AsyncSession,
    *,
    conversation_id: int | None,
    description: str,
    ticket_type: str,
) -> Ticket:
    """写工单 + 会话置「已转人工」。并发撞号（IntegrityError）重算重试一次。"""
    if conversation_id is None:
        raise ValueError("缺少会话上下文，无法创建工单")
    for attempt in (0, 1):
        today = date.today()
        count = (
            await session.execute(
                select(func.count())
                .select_from(Ticket)
                .where(func.date(Ticket.created_at) == today)
            )
        ).scalar_one()
        ticket = Ticket(
            ticket_no=next_ticket_no(count, today),
            conversation_id=conversation_id,
            description=description,
            ticket_type=ticket_type,
        )
        session.add(ticket)
        conv = await session.get(Conversation, conversation_id)
        if conv is not None:
            conv.status = "已转人工"
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            if attempt == 1:
                raise
            logger.warning("ticket_no collision, recomputing (attempt %d)", attempt + 1)
            continue
        await session.refresh(ticket)
        return ticket
    raise RuntimeError("unreachable")


# ---------- ch03: knowledge_chunks / qa_extraction_staging ----------

def chunk_fingerprint(category: str, questions: str, answer: str) -> str:
    import hashlib

    return hashlib.sha1(f"{category}|{questions}|{answer}".encode("utf-8")).hexdigest()


async def add_chunk_drafts(session, drafts, commit: bool = True):
    """pending 入库 + 本批内 prev/next 链(同文档/同批首尾 NULL,§4 尾注)。"""
    rows = [
        KnowledgeChunk(
            category=d.category, questions=d.questions, answer=d.answer,
            section_path=d.section_path, content_type=d.content_type,
            is_key_clause=d.is_key_clause, vectorize_status="pending",
        )
        for d in drafts
    ]
    session.add_all(rows)
    await session.flush()
    for i, r in enumerate(rows):
        r.prev_chunk_id = rows[i - 1].id if i > 0 else None
        r.next_chunk_id = rows[i + 1].id if i + 1 < len(rows) else None
    if commit:
        await session.commit()
    return rows


async def truncate_knowledge_chunks(session) -> None:
    """核对点⑤:自引用 FK 直接 TRUNCATE 报 1701 → session 级开关包裹。活库实测在 Task 9 Step 6。"""
    await session.execute(text("SET FOREIGN_KEY_CHECKS=0"))
    await session.execute(text("TRUNCATE TABLE knowledge_chunks"))
    await session.execute(text("SET FOREIGN_KEY_CHECKS=1"))
    await session.commit()


async def fetch_pending_chunks(session):
    stmt = (
        select(KnowledgeChunk)
        .where(KnowledgeChunk.vectorize_status == "pending")
        .order_by(KnowledgeChunk.id)
    )
    return list((await session.execute(stmt)).scalars().all())


async def fetch_done_ids(session) -> list[int]:
    stmt = select(KnowledgeChunk.id).where(KnowledgeChunk.vectorize_status == "done")
    return [r[0] for r in (await session.execute(stmt)).all()]


async def count_chunks_by_status(session) -> dict[str, int]:
    stmt = select(KnowledgeChunk.vectorize_status, func.count()).group_by(KnowledgeChunk.vectorize_status)
    return {r[0]: r[1] for r in (await session.execute(stmt)).all()}


async def fetch_chunks_by_ids(session, ids: list[int]):
    return list((await session.execute(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids)))).scalars().all())


async def existing_chunk_fingerprints(session) -> set[str]:
    rows = (await session.execute(
        select(KnowledgeChunk.category, KnowledgeChunk.questions, KnowledgeChunk.answer)
    )).all()
    return {chunk_fingerprint(c, q, a) for c, q, a in rows}


async def mark_chunks_vectorized(session, ids: list[int]) -> None:
    """vector_id = str(chunk_id)(§3.1 恒等约定)。demo 规模逐行 update,够用且最直白。"""
    for cid in ids:
        await session.execute(
            update(KnowledgeChunk)
            .where(KnowledgeChunk.id == cid)
            .values(vector_id=str(cid), vectorize_status="done")
        )
    await session.commit()


async def add_qa_staging_rows(session, batch_no: str, source_ref: str, items,
                              status: str = "extracted") -> int:
    session.add_all(
        [
            QaExtractionStaging(batch_no=batch_no, source_ref=source_ref,
                                question=q, answer=a, status=status)
            for q, a in items
        ]
    )
    await session.commit()
    return len(items)


async def reprocess_kept_staging(session) -> int:
    """kept → extracted 整体翻回。仅在「全量重建清空 qa_mined 后找回」场景使用:
    必须先重建库再翻,否则闸3 会撞上自家旧 chunk(cosine=1.0)全军覆没(Task 10 集成测试钉死此顺序)。"""
    res = await session.execute(
        update(QaExtractionStaging).where(QaExtractionStaging.status == "kept").values(status="extracted")
    )
    await session.commit()
    return res.rowcount


async def seen_source_refs(session) -> set[str]:
    rows = (await session.execute(
        select(QaExtractionStaging.source_ref).where(QaExtractionStaging.source_ref.is_not(None))
    )).all()
    return {r[0] for r in rows}


def build_unmined_conversations_query(limit: int):
    """§6 候选:user 与 assistant 都有、且 'conv:{id}' 从未在 staging 出现(会话级记账即幂等)。"""
    has_user = select(Message.conversation_id).where(Message.role == "user")
    has_assistant = select(Message.conversation_id).where(Message.role == "assistant")
    mined = select(QaExtractionStaging.source_ref).where(QaExtractionStaging.source_ref.is_not(None))
    return (
        select(Conversation.id)
        .where(
            Conversation.id.in_(has_user),
            Conversation.id.in_(has_assistant),
            func.concat("conv:", Conversation.id).not_in(mined),
        )
        .order_by(Conversation.id)
        .limit(limit)
    )


async def unmined_conversation_ids(session, limit: int) -> list[int]:
    rows = (await session.execute(build_unmined_conversations_query(limit))).all()
    return [r[0] for r in rows]


async def conversation_transcript(session, conversation_id: int) -> str | None:
    rows = (await session.execute(
        select(Message.role, Message.content)
        .where(Message.conversation_id == conversation_id,
               Message.role.in_(["user", "assistant"]))
        .order_by(Message.id)
    )).all()
    if not rows:
        return None
    return "\n".join(f"{'用户' if role == 'user' else '客服'}: {content}" for role, content in rows)


async def fetch_extracted_staging(session):
    stmt = (
        select(QaExtractionStaging)
        .where(QaExtractionStaging.status == "extracted")
        .order_by(QaExtractionStaging.id)
    )
    return list((await session.execute(stmt)).scalars().all())


async def finalize_qa(session, kept, discarded_ids: list[int]) -> int:
    """单事务:kept 簇 → qa_mined chunk(pending,批内链)+ staging 翻面(§6 幂等论证所在)。"""
    from dataclasses import replace as _dc_replace

    drafts = [_dc_replace(d, content_type="qa_mined") for d, _ in kept]  # 入库统一记 qa_mined(§6)
    await add_chunk_drafts(session, drafts, commit=False)
    if kept:
        kept_ids = [sid for _, ids in kept for sid in ids]
        await session.execute(
            update(QaExtractionStaging).where(QaExtractionStaging.id.in_(kept_ids)).values(status="kept")
        )
    if discarded_ids:
        await session.execute(
            update(QaExtractionStaging).where(QaExtractionStaging.id.in_(discarded_ids)).values(status="discarded")
        )
    await session.commit()
    return len(drafts)


async def clear_staging(session) -> int:
    res = await session.execute(delete(QaExtractionStaging))
    await session.commit()
    return res.rowcount


async def add_low_confidence_question(session, *, conversation_id: int | None,
                                      raw_question: str, source: str,
                                      reason: str | None) -> None:
    """低置信问题池;conversation_id 可空(评估 runner 无会话)。"""
    session.add(LowConfidenceQuestion(conversation_id=conversation_id, raw_question=raw_question,
                                      source=source, reason=reason))
    await session.commit()


# ---- ch04: 原文回查 + 忠实度台账 ----

async def get_chunk(session, chunk_id: int):
    return await session.get(KnowledgeChunk, chunk_id)


async def upsert_faith_case(session, *, eval_id: str, bucket: str, query: str, answer: str,
                            reason: str, citations, judge_model,
                            strategy: str = "hybrid_rerank") -> str:
    """一题一行(uk_eval_id):created/updated/reactivated。复发即退回未解决、清 resolution,
    resolved_at 保留;first_seen_at 不覆写、seen_count+1、重判字段刷新(§3.2/附录A)。"""
    row = (await session.execute(select(FaithCase).where(FaithCase.eval_id == eval_id))).scalar_one_or_none()
    now = datetime.now()
    if row is None:
        session.add(FaithCase(eval_id=eval_id, bucket=bucket, query=query, strategy=strategy,
                              answer=answer, reason=reason, citations=citations, judge_model=judge_model,
                              status="未解决", seen_count=1, first_seen_at=now, last_seen_at=now))
        await session.commit()
        return "created"
    result = "reactivated" if row.status in ("已解决", "无需解决") else "updated"
    if result == "reactivated":
        row.resolution = None
    row.status, row.bucket, row.query, row.strategy = "未解决", bucket, query, strategy
    row.answer, row.reason, row.citations, row.judge_model = answer, reason, citations, judge_model
    row.seen_count += 1
    row.last_seen_at = now
    await session.commit()
    return result


async def list_faith_cases(session, *, status: str | None = None, bucket: str | None = None,
                           limit: int = 200):
    stmt = select(FaithCase).order_by(FaithCase.last_seen_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(FaithCase.status == status)
    if bucket:
        stmt = stmt.where(FaithCase.bucket == bucket)
    return list((await session.execute(stmt)).scalars().all())


async def get_faith_case(session, case_id: int):
    return await session.get(FaithCase, case_id)


async def set_faith_case_status(session, case_id: int, status: str, resolution: str | None):
    row = await session.get(FaithCase, case_id)
    if row is None:
        return None
    row.status = status
    if status == "未解决":
        row.resolution = None  # resolved_at 保留(复发显示)
    else:
        row.resolution = (resolution or "").strip() or None
        row.resolved_at = datetime.now()
    await session.commit()
    return row
