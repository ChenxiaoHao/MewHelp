"""ch07 T6:后台异步摘要任务(spec「后台异步摘要」)。

非阻塞三定死:①schedule 同步返回(create_task,不等);②每 cid 进程内
in-flight 集合防重入(单 worker 语义,多 worker 失效面同 ch06 pending 挂账);
③失败=一条 WARN 即完——不重试不抛穿,下轮超预算自然重触发(压缩丢了不致命,
阻塞才致命)。自开 DB session:请求级 session 关流即还,任务不能寄生其上。
"""

import asyncio
import logging
import time

from langchain_core.messages import HumanMessage
from sqlalchemy import select

from app.context.layers import render_layer2
from app.db import crud
from app.db.models import ConversationSummary
from app.prompts.summary import SUMMARIZE_PROMPT

logger = logging.getLogger(__name__)

_IN_FLIGHT: set[int] = set()


def schedule_summary(session_factory, cid: int, settings, model) -> bool:
    """True=已排后台任务;False=该 cid 任务在飞(summary skip,防重入)。"""
    if cid in _IN_FLIGHT:
        logger.info("summary skip cid=%s (in-flight)", cid)
        return False
    _IN_FLIGHT.add(cid)
    asyncio.create_task(run_summary(session_factory, cid, settings, model))
    return True


async def _load_segments(session, cid) -> list[str]:
    """段表全段按 seq 升序 content(投影重拼料);T6 面读侧查询,不经 crud。"""
    rows = (
        await session.execute(
            select(ConversationSummary.content)
            .where(ConversationSummary.conversation_id == cid)
            .order_by(ConversationSummary.seq)
        )
    ).scalars().all()
    return list(rows)


def _render_batch_text(rows, settings) -> str:
    """层2 半压渲染 → 提示词 batch 位文本(用户:/客服: 行式)。"""
    lines = []
    for m in render_layer2(rows, settings):
        lines.append(f"{'用户' if isinstance(m, HumanMessage) else '客服'}:{m.content}")
    return "\n".join(lines)


def _clean(text: str) -> str:
    """清洗:去首尾空行/空行,>400 字硬截(spec「清洗(去首尾空行、硬上限截断)」)。"""
    body = "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())
    return body[:400]


async def run_summary(session_factory, cid: int, settings, model) -> None:
    t0 = time.monotonic()
    try:
        async with session_factory() as session:
            summary, upto, layer1_from = await crud.get_conv_ctx(session, cid)
            if layer1_from <= upto:
                return  # 层2 无新批(每轮空转防线)
            rows = await crud.list_messages_after(session, cid, upto)
            batch = [r for r in rows if r.id <= layer1_from]
            if not batch:
                return
            rendered = await model.ainvoke(SUMMARIZE_PROMPT.format_messages(
                background=summary or "(无)", batch=_render_batch_text(batch, settings)))
            content = _clean(rendered.content or "")
            if not content:
                logger.info("summary skip cid=%s (空产出)", cid)
                return
            seq = await crud.append_summary_segment(
                session, cid, from_msg_id=upto + 1, upto_msg_id=layer1_from, content=content)
            proj = "\n".join(await _load_segments(session, cid))
            await crud.set_summary_projection(
                session, cid, summary=proj, upto_msg_id=layer1_from)
            logger.info("summary done 第%d段 (%d,%d] 耗时%.2fs",
                        seq, upto, layer1_from, time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001 —— 失败面定死:一条 WARN,不冒穿 event loop
        logger.warning("summary failed cid=%s: %s", cid, exc, exc_info=True)
    finally:
        _IN_FLIGHT.discard(cid)
