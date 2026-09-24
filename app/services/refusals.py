"""统一拒答出口与低置信池写入(spec §5.2/§8)。写池失败只 WARN 不阻断(与 persister 同风格)。"""

from __future__ import annotations

import logging

from app.db import crud
from app.db.engine import get_session_factory

logger = logging.getLogger(__name__)

REFUSAL_ANSWER = (
    "抱歉喵,这个问题我在知识库里没有找到足够可靠的依据,不能凭空作答。"
    "您可以换个说法再问一次,或者让我帮您创建人工工单,由客服跟进处理。"
)


async def pool_low_confidence(conversation_id: int | None, raw_question: str,
                              source: str, reason: str) -> None:
    try:
        async with get_session_factory()() as session:
            await crud.add_low_confidence_question(
                session, conversation_id=conversation_id, raw_question=raw_question,
                source=source, reason=reason,
            )
    except Exception:  # noqa: BLE001 —— 池是复盘材料,丢一行不配打断用户
        logger.warning("低置信池写入失败(不阻断拒答): %s", raw_question, exc_info=True)
