"""ch10 T9 旁路批量归类 job:攒低置信度池未归类行 → 微调分类器 → 落 topic_classifications。

用法:uv run python -m app.jobs.topic_classify [--limit N] [--rerun] [--dry-run]
- 默认只处理未归类行(LEFT JOIN IS NULL);--rerun 处理全池并 upsert 刷新;
- uk_question_id 一人一行,重跑幂等;Langfuse 落一条 trace(name=ch10_topic_classify)。
实时对话主链路不写不读本表。GBK 红线:控制台 ASCII。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import delete, insert, select, update  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.engine import dispose_engine, get_session_factory, init_engine  # noqa: E402
from app.db.models import LowConfidenceQuestion, TopicClassification  # noqa: E402

BATCH = 32


async def collect_unclassified(session, limit: int | None,
                               rerun: bool) -> list[tuple[int, str]]:
    q = (select(LowConfidenceQuestion.id, LowConfidenceQuestion.raw_question)
         .outerjoin(TopicClassification,
                    TopicClassification.question_id == LowConfidenceQuestion.id)
         .where(TopicClassification.id.is_(None) if not rerun else True)
         .order_by(LowConfidenceQuestion.id))
    if limit:
        q = q.limit(limit)
    return [(r[0], r[1]) for r in (await session.execute(q)).all()]


async def store_results(session, pairs: list[tuple[int, list[str]]],
                        rerun: bool) -> int:
    """未归类行 insert;--rerun 时已存在行 update labels。返回处理行数。"""
    n = 0
    for qid, labels in pairs:
        existing = (await session.execute(
            select(TopicClassification.id)
            .where(TopicClassification.question_id == qid))).scalar()
        if existing is None:
            await session.execute(insert(TopicClassification).values(
                question_id=qid, labels=labels))
        elif rerun:
            await session.execute(
                update(TopicClassification)
                .where(TopicClassification.id == existing)
                .values(labels=labels))
        n += 1
    await session.commit()
    return n


def _trace(st, processed: int, args) -> None:
    try:
        from langfuse import Langfuse
        lf = Langfuse(host=st.langfuse_host, public_key=st.langfuse_public_key,
                      secret_key=st.langfuse_secret_key)
        t = lf.trace(name="ch10_topic_classify", tags=["ch10"],
                     input={"limit": args.limit, "rerun": args.rerun},
                     output={"processed": processed})
        lf.flush()
    except Exception:  # noqa: BLE001 —— trace 失败不拦批处理,显式打出
        print("[topic_classify] WARN langfuse trace skipped", flush=True)


async def _amain(args) -> int:
    st = get_settings()
    init_engine(st)
    try:
        from app.services.topic_classifier import TopicClassifier

        async with get_session_factory()() as session:
            rows = await collect_unclassified(session, args.limit, args.rerun)
            print(f"[topic_classify] selected={len(rows)} rerun={args.rerun}",
                  flush=True)
            if args.dry_run or not rows:
                return 0
            clf = TopicClassifier()
            n = 0
            for i in range(0, len(rows), BATCH):
                chunk = rows[i:i + BATCH]
                from finetune.clean import clean  # 训练面同口径(终审 I-4)
                labels = clf.classify_batch([clean(t) for _, t in chunk])
                n += await store_results(session,
                                         [(qid, lb) for (qid, _), lb in
                                          zip(chunk, labels)], args.rerun)
                print(f"[topic_classify] processed={n}/{len(rows)}", flush=True)
            _trace(st, n, args)
        return 0
    finally:
        await dispose_engine()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="topic_classify")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return asyncio.run(_amain(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
