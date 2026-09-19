"""提取 Prompt 评估集：跑真实模型对比人工标注。

用法: uv run python evals/run_extraction_eval.py
判定: order_id / issue_type 精确匹配; expected_solution 关键词包含 + 打印人工复核。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.services.chat_service import get_model
from app.services.extract_service import extract_after_sale


async def main() -> int:
    samples = json.loads(
        (Path(__file__).parent / "extraction_samples.json").read_text(encoding="utf-8")
    )
    model = get_model(get_settings())
    passed = 0
    for i, s in enumerate(samples, 1):
        got = await extract_after_sale(s["description"], model)
        exp = s["expected"]
        ok_id = got.order_id == exp["order_id"]
        ok_type = got.issue_type.value == exp["issue_type"]
        ok_sol = any(
            k in got.expected_solution for k in exp["expected_solution_keywords"]
        )
        ok = ok_id and ok_type and ok_sol
        passed += ok
        print(
            f"[{'PASS' if ok else 'FAIL'}] #{i} "
            f"order_id: {got.order_id!r}(期望 {exp['order_id']!r}) | "
            f"issue_type: {got.issue_type.value}(期望 {exp['issue_type']}) | "
            f"expected_solution: {got.expected_solution!r}"
            f"(关键词 {exp['expected_solution_keywords']})"
        )
    print(f"\n通过率: {passed}/{len(samples)}")
    return 0 if passed == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
