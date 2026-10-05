"""飞轮补扫 CLI(拍板 2A):扫 matched_review_id IS NULL 重跑流水线。

uv run python -m app.jobs.flywheel            # 补扫全部未处理行
uv run python -m app.jobs.flywheel --limit 20 # 本轮最多 20 行
uv run python -m app.jobs.flywheel --dry-run  # 只列未处理行数与 id,不动 LLM 不写库

三态通用:进程崩/fire-and-forget 丢失/批量回填。幂等=matched 写回即出 NULL 集
(Focus 6 单事务封口保证「半崩=两边都不存在」)。控制台 ASCII-only(GBK 红线)。
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.core.config import get_settings
from app.db import crud
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.services import flywheel


async def _run(args: argparse.Namespace) -> int:
    try:
        async with get_session_factory()() as session:
            ids = await crud.unprocessed_lcq_ids(session, limit=args.limit)
        print(f"[flywheel] unprocessed rows: {len(ids)} ids={ids[:200]}")
        if args.dry_run:
            return 0
        for rid in ids:
            await flywheel.process_lcq_row(rid)  # 内部全吞:单行崩不拖批次
        print(f"[flywheel] swept {len(ids)} rows")
        return 0
    finally:
        await dispose_engine()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flywheel")
    parser.add_argument("--limit", type=int, default=None,
                        help="本轮最多处理行数(默认不限)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只列未处理行,不动 LLM 不写库")
    args = parser.parse_args(argv)
    init_engine(get_settings())
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
