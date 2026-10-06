"""ch10 T3 清洗:脱敏+格式规范(确定性代码面)。错别字归一在 T5 预标调用(拍板 2)。

CLI(样例验证面):uv run python -m finetune.clean --dump finetune/audit/pool_cleaned.csv
把真池 138 条 raw_question 清洗后落 UTF-8(-sig)CSV 供人工查。控制台只出 ASCII。
"""

import argparse
import csv
import re
import sys
from pathlib import Path

_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_LONG_NUM = re.compile(r"(?<!\d)\d{10,}(?!\d)")
_ADDR = re.compile(
    r"[^，,。;；\s]{2,}(?:省|市|区|县|镇|乡|村)"
    r"[^，,。;；\s]{0,20}?(?:路|街|道|号|栋|幢|单元|室|楼)[^，,。;；\s]{0,15}"
)

_FW = {i: i - 0xFEE0 for i in range(0xFF01, 0xFF5F)}  # 全角 ASCII → 半角
_FW[0x3000] = 0x20


def desensitize(text: str) -> str:
    text = _PHONE.sub("<PHONE>", text)
    text = _ADDR.sub("<ADDR>", text)
    return _LONG_NUM.sub("<NUM>", text)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_FW)).strip()


def clean(text: str) -> str:
    return normalize(desensitize(normalize(text)))


async def _dump(path: Path) -> int:
    from app.core.config import get_settings
    from app.db.engine import dispose_engine, get_session_factory, init_engine
    from app.db.models import LowConfidenceQuestion

    init_engine(get_settings())
    try:
        async with get_session_factory()() as s:
            from sqlalchemy import select
            rows = (await s.execute(
                select(LowConfidenceQuestion.id, LowConfidenceQuestion.raw_question)
                .order_by(LowConfidenceQuestion.id))).all()
    finally:
        await dispose_engine()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "raw", "cleaned"])
        for rid, raw in rows:
            w.writerow([rid, raw, clean(raw)])
    return len(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="clean")
    p.add_argument("--dump", type=Path, default=None)
    args = p.parse_args(argv)
    if args.dump:
        import asyncio
        n = asyncio.run(_dump(args.dump))
        print(f"[clean] dumped rows={n} -> {args.dump.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
