"""ch07 T5 摘要 prompt 质量冒烟(P7 豁免口径:纯 Prompt 任务用标注样例代 TDD)。

uv run python evals/smoke_ch07.py
—— 载 tests/samples/ch07_summary_qa.csv,SUMMARIZE_PROMPT 真模型直调,逐行打印:
包含判定(expected_contains any-of)+ 长度判定(≤max_chars)+ 禁编造 spot-check
(梗概里 ≥4 位数字串必须能在本批原文找到)。全行通过才算过;miss → 只动
铁律措辞/few-shot,不动代码(ch06 纪律)。恒 exit 0,不进 CI。
"""

import asyncio
import csv
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Windows GBK 控制台防崩:✓/✗/中文全走 UTF-8(红线)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.core.config import get_settings  # noqa: E402
from app.prompts.summary import SUMMARIZE_PROMPT  # noqa: E402
from app.services.chat_service import get_model  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "samples"


def _fabricated_ids(out: str, batch: str) -> list[str]:
    """梗概中 ≥4 位数字串,凡本批原文没有的即编造嫌疑。"""
    return [d for d in re.findall(r"\d{4,}", out) if d not in batch]


async def run_one(model, row) -> bool:
    msgs = SUMMARIZE_PROMPT.format_messages(
        background=row["background"] or "(无)", batch=row["batch"])
    out = (await model.ainvoke(msgs)).content.strip()
    cands = [c for c in row["expected_contains"].split("|") if c]
    hit = any(c in out for c in cands)
    cap = int(row["max_chars"])
    fit = len(out) <= cap
    fake = _fabricated_ids(out, row["batch"])
    ok = hit and fit and not fake
    print(f"[{'✓' if ok else '✗'}] batch={row['batch']!r}")
    print(f"    梗概({len(out)}字): {out!r}")
    print(f"    含{'/'.join(cands)}:{hit}  ≤{cap}:{fit}  编造数字:{fake or '无'}")
    return ok


async def main() -> None:
    model = get_model(get_settings())
    with open(SAMPLES / "ch07_summary_qa.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    results = [await run_one(model, row) for row in rows]
    print(f"\n摘要 smoke:{sum(results)}/{len(results)} 行通过(要求全过)")


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
