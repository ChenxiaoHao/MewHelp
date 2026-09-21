"""建库 CLI(spec §9)。

uv run python -m app.jobs.build_knowledge                # 全量重建(含 qa_mined 清空,§5 副作用)
uv run python -m app.jobs.build_knowledge --skip-existing # 保留现有 + 捡 pending(验收2 重跑用)
uv run python -m app.jobs.build_knowledge --fault-after 3 # 向量化注入崩溃(验收2)
uv run python -m app.jobs.build_knowledge --check         # 双端对账
"""

import argparse
import asyncio
import sys

from app.core.config import get_settings
from app.db.engine import dispose_engine, init_engine
from app.rag import indexer


async def _run(args: argparse.Namespace) -> int:
    try:
        if args.check:
            return await indexer.check()
        await indexer.ingest_docs("knowledge", skip_existing=args.skip_existing)
        await indexer.vectorize_pending(fault_after=args.fault_after)
        return 0
    finally:
        await dispose_engine()  # 与 init 同一 asyncio.run 生命周期内释放


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_knowledge")
    parser.add_argument("--skip-existing", action="store_true",
                        help="不清表:指纹跳过已有块并继续补齐 pending 向量(§0-8)")
    parser.add_argument("--fault-after", type=int, default=None, metavar="N",
                        help="向量化满 N 块(批粒度)后 exit 42,崩溃演练")
    parser.add_argument("--check", action="store_true", help="pending/done/Milvus 双端对账")
    args = parser.parse_args(argv)
    init_engine(get_settings())
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
