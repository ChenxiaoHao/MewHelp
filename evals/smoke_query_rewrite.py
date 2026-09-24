"""查询理解标注样例冒烟(spec §9:prompt 类任务不套 TDD):6 道老师题库原题手工核对改写质量。
uv run python evals/smoke_query_rewrite.py —— 输出逐行 JSON,人工判「关键信息丢没丢、同义词是不是实词」,
结论与样例输出贴进 dev-notes;不达标 → 只改 prompt 再跑,不改断言。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.rag.query_understanding import understand_query  # noqa: E402

SAMPLES = [
    "东西不想要了还能退不",
    "钱退给我要等到什么时候啊",
    "我在新疆下单 80 块钱,运费怎么算,会员的免运费能不能抵",
    "MH-LP100 的废砂盒多久倒一次",
    "夜里十一点客服还在线吗",
    "猫粮拆封了还能退吗",
]


async def main() -> None:
    st = get_settings()
    for q in SAMPLES:
        r = await understand_query(q, st)
        print(json.dumps({"原话": q, "标准问法": r.standard_query,
                          "同义词": r.synonyms, "降级": r.degraded}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
