"""工具路由评估（Prompt 类产出的评估集替代 TDD，spec §10）。

真实模型调用：SYSTEM_PROMPT + 每条样例 → bind_tools → 第一轮 ainvoke
（只看决策，不执行工具，零 DB 依赖）→ 比对工具名集合。
用法：uv run python evals/run_tool_routing_eval.py   （需 .env 有效 OPENAI_* 配置）
判定：≥8/10 过阈值（目标 10/10）；FAIL 样例逐条分析原因并记录 dev-notes。
确定性：评估固定 temperature=0（生产聊天仍用 settings.temperature=0.7）——
路由决策可复现，阈值判定才有意义（2026-09-20 与用户确认）。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.prompts.customer_service import SYSTEM_PROMPT  # noqa: E402
from app.services.chat_service import get_model  # noqa: E402
from app.tools.registry import get_tools  # noqa: E402

PASS_THRESHOLD = 8


async def main() -> int:
    samples = json.loads(
        (Path(__file__).parent / "tool_routing_samples.json").read_text(encoding="utf-8")
    )
    eval_settings = get_settings().model_copy(update={"temperature": 0})
    model = get_model(eval_settings).bind_tools(get_tools())

    passed = 0
    for s in samples:
        resp = await model.ainvoke(
            [("system", SYSTEM_PROMPT), HumanMessage(content=s["question"])]
        )
        actual = sorted(tc["name"] for tc in (getattr(resp, "tool_calls", None) or []))
        expected = sorted(s["expected_tools"])
        ok = actual == expected
        passed += ok
        mark = "PASS" if ok else "FAIL"
        remark = f" | remark: {s['remark']}" if s.get("remark") else ""
        print(f"[{mark}] #{s['id']:>2} {s['question']} | 期望 {expected} 实际 {actual}{remark}")

    total = len(samples)
    print(f"\n通过率: {passed}/{total}（阈值 {PASS_THRESHOLD}/{total}）")
    if passed < PASS_THRESHOLD:
        print("未达阈值：分析 FAIL 样例（Prompt 指引问题 or 模型能力问题），调整后重跑。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
