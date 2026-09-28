"""ch06 prompt 质量冒烟(spec「Prompt 质量」节:prompt 类任务不套 TDD,标注样例替代)。

uv run python evals/smoke_ch06.py
—— 载入 tests/samples/ch06_intent_qa.csv / ch06_coref_qa.csv,真模型直调
INTENT_PROMPT(parse_intent_json 判定)/ COREF_PROMPT(expected_contains any-of 包含),
打印逐行对照+准确率;恒 exit 0,不进 CI。不达标(<85%)→ 只调 few-shot 表述重跑,
不动 route 表/代码(选型定死)。coref 原始输出逐行打印,顺带定夺 M1-F2
(模型是否稳定单行输出 → 节点取末行 vs spec「首行」)。
"""

import asyncio
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Windows GBK 控制台防崩:✓/✗/中文全走 UTF-8,不可映射字符 replace
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.core.config import get_settings  # noqa: E402
from app.prompts.coref import COREF_PROMPT  # noqa: E402
from app.prompts.intent import INTENT_PROMPT  # noqa: E402
from app.services.chat_service import get_model  # noqa: E402
from app.workflows.routing import INTENTS, parse_intent_json  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "samples"


async def run_intent(model) -> float:
    ok = total = 0
    with open(SAMPLES / "ch06_intent_qa.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            raw = (await model.ainvoke(
                INTENT_PROMPT.format_messages(question=row["question"]))).content
            parsed = parse_intent_json(raw)
            got = parsed[0] if parsed else f"PARSE-FAIL({raw[:24]!r})"
            hit = got == row["expected_intent"]
            ok += hit
            print(f"[intent {'✓' if hit else '✗'}] {row['question']}"
                  f" → 期望:{row['expected_intent']} 实际:{got} conf:{parsed[1] if parsed else '-'}")
    return ok / total


async def run_coref(model) -> float:
    ok = total = 0
    with open(SAMPLES / "ch06_coref_qa.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            history = row["history"] or "(无)"
            raw = (await model.ainvoke(COREF_PROMPT.format_messages(
                history=history, question=row["question"]))).content.strip()
            cands = [c for c in row["expected_contains"].split("|") if c]
            hit = any(c in raw for c in cands)
            ok += hit
            print(f"[coref {'✓' if hit else '✗'}] hist={history!r} q={row['question']!r}"
                  f" → {raw!r} (期望含 {'/'.join(cands)}) 行数:{len(raw.splitlines())}")
    return ok / total


async def main() -> None:
    model = get_model(get_settings())
    acc_i = await run_intent(model)
    acc_c = await run_coref(model)
    print(f"\n意图准确率 {acc_i:.0%}  (阈值 85%)")
    print(f"消解准确率 {acc_c:.0%}  (阈值 85%)")


if __name__ == "__main__":
    print(f"八类枚举核对: {INTENTS}")
    asyncio.run(main())
    sys.exit(0)
