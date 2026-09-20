"""数据访问层：会话/消息/FAQ 检索/工单。运行期落库失败的降级策略在调用方（persister）。"""

import logging
from datetime import date

from sqlalchemy import Select, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Faq, Message, Ticket

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
