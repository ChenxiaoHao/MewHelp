"""qa_mining prompt 评估(替代 TDD 的数据任务质量门)。uv run python evals/run_qa_mining_eval.py"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目惯例:从 evals/ 目录跑也能 import app

from app.jobs.mine_qa import _build_model, mining_messages  # noqa: E402
from app.schemas.qa_mining import MinedQA  # noqa: E402


async def main_async() -> int:
    samples = json.loads(Path("evals/qa_mining_samples.json").read_text(encoding="utf-8"))
    structured = _build_model()
    bad = []
    for s in samples:
        result: MinedQA = await structured.ainvoke(mining_messages(s["dialog"]))
        allq = "\n".join(i.question for i in result.items)
        alla = "\n".join(i.answer for i in result.items)
        reasons = []
        if "expect_items_max" in s and len(result.items) > s["expect_items_max"]:
            reasons.append(f"多余抽取 {len(result.items)} 条")
        if s["expect_question_any"] and not any(k in allq for k in s["expect_question_any"]):
            reasons.append("漏抽(问法关键词未出现)")
        for k in s["expect_answer_all"]:
            if k not in alla:
                reasons.append(f"答案缺关键内容: {k}")
        for f in s["hallucination_forbidden"]:
            if f in alla or f in allq:
                reasons.append(f"幻觉: {f}")
        print(f"[{'PASS' if not reasons else 'FAIL'}] {s['id']}: {';'.join(reasons) or 'OK'} (抽 {len(result.items)} 条)")
        if reasons:
            bad.append(s["id"])
    print(f"\nqa_mining eval: {len(samples) - len(bad)}/{len(samples)}")
    return 0 if len(samples) - len(bad) >= len(samples) - 1 else 1  # 5 条最多容忍 1 条失败


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
