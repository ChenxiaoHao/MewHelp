"""ch05 Task 7: 建单与「转人工」解耦(D4)——建单后 Conversation.status 不变。

唯一 DB 语义改动:crud.create_ticket 移除 conv.status='已转人工' 副作用;
tickets 行、编号格式、撞号重试、conversation_id 连带绑定全部不动。
真库往返沿用 ledger_integration 同款 engine 夹具模式(integration 标记,
默认 deselect,演示/验收时 -m integration 显式跑)。
"""

import re
import uuid

import pytest
from sqlalchemy import delete, select

from app.core.config import get_settings
from app.db import crud
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.db.models import Conversation, Ticket

pytestmark = pytest.mark.integration


@pytest.fixture
async def session():
    init_engine(get_settings())
    async with get_session_factory()() as s:
        yield s
    await dispose_engine()


async def _mk_conv(s, tag):
    conv = await crud.create_or_get_conversation(s, None, f"u_t7_{tag}")
    await s.commit()
    return conv


async def _status_of(s, conv_id):
    return (await s.execute(
        select(Conversation.status).where(Conversation.id == conv_id)
    )).scalar_one()


async def _cleanup(s, conv_id):
    await s.execute(delete(Ticket).where(Ticket.conversation_id == conv_id))
    await s.execute(delete(Conversation).where(Conversation.id == conv_id))
    await s.commit()


async def test_create_ticket_keeps_conversation_status(session):
    """行为变更主测:建单前后 status 快照逐字符相等(旧代码此处必见「已转人工」)。"""
    tag = uuid.uuid4().hex[:8]
    conv = await _mk_conv(session, tag)
    before = await _status_of(session, conv.id)
    await crud.create_ticket(session, conversation_id=conv.id,
                             description="退款被拒想人工", ticket_type="售后")
    after = await _status_of(session, conv.id)
    assert after == before
    await _cleanup(session, conv.id)


async def test_ticket_row_and_no_format_unchanged(session):
    """回归钉:行仍落库、编号格式 T{YYYYMMDD}{NNN}、conversation_id 连带绑定不动。"""
    tag = uuid.uuid4().hex[:8]
    conv = await _mk_conv(session, tag)
    t = await crud.create_ticket(session, conversation_id=conv.id,
                                 description="T7回归", ticket_type="咨询")
    assert re.fullmatch(r"T\d{11}", t.ticket_no)  # T+YYYYMMDD(8)+序号(3)
    row = (await session.execute(
        select(Ticket).where(Ticket.ticket_no == t.ticket_no)
    )).scalar_one()
    assert row.conversation_id == conv.id and row.description == "T7回归"
    await _cleanup(session, conv.id)


async def test_multiple_tickets_same_conversation(session):
    """两会话各建多单合法(解耦不引入任何新约束)。"""
    tag = uuid.uuid4().hex[:8]
    conv = await _mk_conv(session, tag)
    t1 = await crud.create_ticket(session, conversation_id=conv.id,
                                  description="单A", ticket_type="咨询")
    t2 = await crud.create_ticket(session, conversation_id=conv.id,
                                  description="单B", ticket_type="投诉")
    assert t1.ticket_no != t2.ticket_no
    assert await _status_of(session, conv.id) == "进行中"  # 模型默认值,多单不动
    await _cleanup(session, conv.id)
