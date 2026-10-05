"""ch09 T8 评估:飞轮标准化/查重两提示真模型跑批(标注样例集,plan「纯 Prompt 任务」验收面)。

真模型直调 flywheel.normalize_llm / dedup_llm(与线同一 _structured 路:
get_model 同源配置 + temperature=0)——测的就是当前 few-shot 的质量,调优只动
app/prompts/flywheel.py 示例,不动本脚本与代码路。

判定:脚本机判辅助(过线 ≥8/10),中文样例逐条人工核对为准——明细落
evals/reports/ch09_flywheel_eval_result.json(UTF-8),控制台只出 ASCII 摘要
(GBK 红线)。dedup 样例自带 norm(与标准化质量解耦,单测查重判定面)。

用法: uv run python evals/run_ch09_flywheel_eval.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.schemas.flywheel import NormalizedQA  # noqa: E402
from app.services import flywheel  # noqa: E402

PASS_THRESHOLD = 8


def judge_normalize(s: dict, out) -> tuple[bool, dict]:
    nq = (out.normalized_question or "").strip()
    sa = (out.suggested_answer or "").strip()
    missing = [g for g in s["expect_any"] if not any(k in nq for k in g)]
    sig = {"normalized_question": nq, "suggested_answer": sa,
           "missing_groups": missing}
    return bool(nq) and bool(sa) and not missing, sig


def judge_dedup(s: dict, matched) -> tuple[bool, dict]:
    sig = {"matched_id": matched, "expect": s["expect_matched"]}
    return matched == s["expect_matched"], sig


async def run_sample(s: dict, st) -> tuple[bool, dict]:
    if s["kind"] == "normalize":
        out = await flywheel.normalize_llm(s["raw"], st)
        return judge_normalize(s, out)
    norm = NormalizedQA(normalized_question=s["norm"], suggested_answer="")
    matched = await flywheel.dedup_llm(
        s["raw"], norm, [tuple(c) for c in s["candidates"]], st)
    return judge_dedup(s, matched)


async def main() -> int:
    samples = [json.loads(ln) for ln in
               (Path(__file__).parent / "flywheel_samples.jsonl").read_text(
                   encoding="utf-8").splitlines() if ln.strip()]
    st = get_settings()
    results = []
    n_pass = 0
    for s in samples:
        ok, sig = await run_sample(s, st)
        n_pass += ok
        results.append({"id": s["id"], "kind": s["kind"], "raw": s["raw"],
                        "verdict": "PASS" if ok else "FAIL",
                        "notes": s["notes"], "signals": sig})
        print(f"sample {s['id']:>2} [{s['kind']:<9}] {'PASS' if ok else 'FAIL'}")

    out = Path(__file__).parent / "reports" / "ch09_flywheel_eval_result.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"threshold": PASS_THRESHOLD, "passed": n_pass, "total": len(samples),
         "overall": "PASS" if n_pass >= PASS_THRESHOLD else "FAIL",
         "samples": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TOTAL {n_pass}/{len(samples)} (threshold {PASS_THRESHOLD}) -> "
          f"evals/reports/ch09_flywheel_eval_result.json")
    return 0 if n_pass >= PASS_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
