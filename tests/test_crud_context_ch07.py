"""ch07 T2 crud 六函数活库往返(spec「持久化与数据访问」/brief 接口面)。

锚点/段表/列表查询全是真 SQL 语义(NULL→0、seq=MAX+1 撞 uk 重试、id 边界过滤、
首问预览),FakeSession 断不到查询计划——整组按 brief 挂 integration 跑 dev 活库,
每次用随机 user_id 自清理(ledger integration 同款纪律)。默认套件被 -m 过滤,
手动执行:uv run pytest tests/test_crud_context_ch07.py -m integration -q。
"""

import uuid

import pytest
from sqlalchemy import delete, select

from app.core.config import get_settings
from app.db import crud
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.db.models import Conversation, ConversationSummary, Message

pytestmark = pytest.mark.integration


async def _roundtrip():
    init_engine(get_settings())
    uid = f"it-ch07-{uuid.uuid4().hex[:12]}"
    try:
        async with get_session_factory()() as session:
            conv = await crud.create_or_get_conversation(session, None, uid)
            cid = conv.id

            # 空锚点:NULL 一律按 0(新会话全史在层1)
            assert await crud.get_conv_ctx(session, cid) == (None, 0, 0)

            for i in range(4):
                await crud.add_message(session, cid, "user", f"第{i}问{'汉' * 60}")
                await crud.add_message(session, cid, "assistant", f"第{i}答")
            ids = [r[0] for r in (await session.execute(
                select(Message.id).where(Message.conversation_id == cid).order_by(Message.id)
            )).all()]
            assert len(ids) == 8

            # 回载:after_id 之后升序,含 tool 行口径由 T9 消费,这里钉边界与排序
            after = await crud.list_messages_after(session, cid, ids[1])
            assert [m.id for m in after] == ids[2:]

            # 层1锚点直写
            await crud.set_layer1_from(session, cid, ids[3])
            assert await crud.get_conv_ctx(session, cid) == (None, 0, ids[3])

            # 段表追加:seq 从 1 起连续
            s1 = await crud.append_summary_segment(
                session, cid, from_msg_id=ids[0], upto_msg_id=ids[3], content="第一段梗概")
            s2 = await crud.append_summary_segment(
                session, cid, from_msg_id=ids[4], upto_msg_id=ids[7], content="第二段梗概")
            assert (s1, s2) == (1, 2)

            # 投影列:拼装位 + 边界追平
            await crud.set_summary_projection(
                session, cid, summary="第一段梗概\n第二段梗概", upto_msg_id=ids[7])
            summary, upto, l1 = await crud.get_conv_ctx(session, cid)
            assert summary == "第一段梗概\n第二段梗概" and upto == ids[7] and l1 == ids[3]

            # 列表:新在前、首问预览截 40 字、已摘要标记
            rows = await crud.list_user_conversations(session, uid)
            assert len(rows) == 1
            rid, rcreated, preview, summarized = rows[0]
            assert rid == cid and rcreated is not None and summarized is True
            assert preview.startswith("第0问") and len(preview) <= 40
    finally:
        async with get_session_factory()() as session:
            conv_ids = select(Conversation.id).where(Conversation.user_id == uid)
            await session.execute(delete(ConversationSummary).where(
                ConversationSummary.conversation_id.in_(conv_ids)))
            await session.execute(delete(Message).where(Message.conversation_id.in_(conv_ids)))
            await session.execute(delete(Conversation).where(Conversation.user_id == uid))
            await session.commit()
        await dispose_engine()


async def test_crud_context_roundtrip_live():
    await _roundtrip()
