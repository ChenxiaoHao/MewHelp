"""ch09 飞轮流水线:LCQ 行 → 标准化 → 查重 → 归并/新建 review_queue(spec「飞轮流水线」节)。

process_lcq_row 异常全吞只 WARN——fire-and-forget 任务无人接收,行留
matched_review_id NULL 态由 CLI 补扫兜底(拍板 2A)。幂等边界=入口检查
matched 非 NULL 即返;Focus 6 封口在 crud.merge_lcq_into_queue 单事务。
"""

from __future__ import annotations

import asyncio
import logging

from app.core.config import get_settings
from app.db import crud
from app.db.engine import get_session_factory
from app.prompts.flywheel import dedup_messages, normalize_messages
from app.schemas.flywheel import DedupMatch, NormalizedQA

logger = logging.getLogger(__name__)

CANDIDATE_CAP = 50  # 查重候选窗:待审全量≤50,超出按 updated_at 最新截(spec)

# create_task 裸引用可能被 GC 半路蒸发(asyncio 已知陷阱)——模块级强引用集。
_BG: set = set()


def _structured(schema, st):
    """get_model 同源配置 + temperature=0 + 结构化输出(mine_qa._build_model 同形)。"""
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=st.model_name, temperature=0,
                       api_key=st.openai_api_key, base_url=st.openai_base_url)
    return model.with_structured_output(schema)


async def normalize_llm(raw_question: str, settings) -> NormalizedQA:
    return await _structured(NormalizedQA, settings).ainvoke(
        normalize_messages(raw_question))


async def dedup_llm(raw_question: str, norm: NormalizedQA,
                    candidates: list[tuple[int, str]], settings) -> int | None:
    r = await _structured(DedupMatch, settings).ainvoke(
        dedup_messages(raw_question, norm.normalized_question, candidates))
    return r.matched_id


def spawn_process(row_id: int) -> None:
    """落池成功后的 fire-and-forget 触发(拍板 2A)。触发自身失败=行留 NULL
    等补扫,不得反噬已成功的池写/已应答的用户请求——异常全吞只 WARN。"""
    coro = process_lcq_row(row_id)
    try:
        task = asyncio.create_task(coro)
    except Exception:  # noqa: BLE001 —— 无运行 loop 等
        coro.close()  # create_task 未消费协程时手动收尾,免 never-awaited 噪声
        logger.warning("flywheel spawn failed; row left for sweep id=%s",
                       row_id, exc_info=True)
        return
    _BG.add(task)
    task.add_done_callback(_BG.discard)


async def process_lcq_row(row_id: int, *, settings=None, normalizer=None,
                          deduper=None) -> None:
    """标准化→查重→入队/归并+matched 写回。任何失败=行留 NULL 不外泄。

    normalizer/deduper 注入位:T8 真模型跑批走默认 LLM 路;活库测免网跑。
    """
    try:
        st = settings or get_settings()
        norm_fn = normalizer or normalize_llm
        dedup_fn = deduper or dedup_llm
        async with get_session_factory()() as session:
            row = await crud.get_lcq_row(session, row_id)
            if row is None or row.matched_review_id is not None:
                return  # 行不存在/已处理:幂等边界,不吃第二遍
            raw = row.raw_question
            cands, total = await crud.pending_review_candidates(
                session, cap=CANDIDATE_CAP)
        norm = await norm_fn(raw, st)
        nq = (norm.normalized_question or "").strip()
        if not nq:
            raise ValueError(f"normalized_question 为空 id={row_id}")
        matched = await dedup_fn(raw, norm, cands, st)
        if matched is not None and matched not in {c[0] for c in cands}:
            logger.warning("flywheel: dedup matched=%s 不在候选集,按未命中处理 id=%s",
                           matched, row_id)  # LLM 幻觉 id 不得指向他行累加
            matched = None
        if total > CANDIDATE_CAP:
            logger.warning(
                "flywheel: pending queue total=%s > cap=%s, dedup truncated to "
                "newest window id=%s", total, CANDIDATE_CAP, row_id)
        async with get_session_factory()() as session:
            await crud.merge_lcq_into_queue(
                session, row_id, matched_id=matched,
                normalized_question=nq[:500],  # VARCHAR(512) 列宽余量,防超长崩成毒行
                suggested_answer=norm.suggested_answer)
    except Exception:  # noqa: BLE001 —— spec:任务内异常全吞→WARN
        logger.warning("flywheel 行处理失败(留 NULL 待补扫) id=%s", row_id,
                       exc_info=True)
