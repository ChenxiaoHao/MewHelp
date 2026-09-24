"""ch04 验收演练(README 引用)。前置:docker mysql/milvus 起、.env key 配好、build_knowledge 跑过。
验收①=run_strategy_eval 报告;验收③的 UI 侧在聊天页人工点(T13)。"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.engine import dispose_engine, get_session_factory, init_engine  # noqa: E402
from app.db.models import LowConfidenceQuestion  # noqa: E402
from app.rag import retriever  # noqa: E402
from app.rag.hit_format import format_hits  # noqa: E402


async def main() -> None:
    st = get_settings()
    init_engine(st)
    try:
        r = await retriever.retrieve("MH-LP100 猫粮保质期多久", strategy="bm25", settings=st)
        print(f"[验收2] BM25 型号题 top1 → {r.chunks[0].row.section_path}")
        h = await retriever.retrieve("退货要寄回去吗,邮费谁出", strategy="hybrid_rerank", settings=st)
        for x in format_hits(retriever.apply_head_tail(h.chunks))[:3]:  # 与 query_faq 同排布,[n] 编号才与生产一致
            print(f"[验收3] 引用[{x['n']}] chunk_id={x['id']} → {x['section_path']}")
        g = await retriever.retrieve("我家猫想吃没有的东西", strategy="hybrid_rerank", settings=st)
        print(f"[验收4] refused={g.refused} note={g.note}")
        async with get_session_factory()() as session:
            n = (await session.execute(
                select(func.count()).select_from(LowConfidenceQuestion))).scalar()
        print(f"[验收4] low_confidence_questions 池现有 {n} 行(UI 问同类问题后此数 +1)")
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
