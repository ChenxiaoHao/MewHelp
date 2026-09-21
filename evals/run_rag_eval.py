"""RAG 检索评估(§9:替代 TDD 的验收数据门)。通过线:hit-rate@3 ≥10/12 且运费组(1-5)5/5。

confusable 规则(spec 附录 C):对偶块与期望块同现 top-3 时,对偶不得排在期望之前。
前置:docker compose up + build_knowledge 完成 + mine_qa 尚未跑。
uv run python evals/run_rag_eval.py [--top-k 3] [--debug]   # --debug 临时阈值 0,看真实分数分布
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402 项目惯例:从 evals/ 目录跑也能 import app

from app.core.config import get_settings
from app.db.engine import dispose_engine, init_engine
from app.rag import retriever

PASS_MIN = 10


async def _main(samples, top_k) -> int:
    bad = []
    for s in samples:
        hits = await retriever.retrieve_hits(s["query"])
        paths = [h.section_path or "" for h in hits][:top_k]
        hit_rank = next((i for i, p in enumerate(paths) if s["expect_path"] in p), None)
        ok = hit_rank is not None
        crossed = []
        for rid in s.get("confusable_with", []):
            rival = next(x for x in samples if x["id"] == rid)
            rival_rank = next((i for i, p in enumerate(paths) if rival["expect_path"] in p), None)
            if ok and rival_rank is not None and rival_rank < hit_rank:
                crossed.append(str(rid))
        if crossed:
            ok = False
        mark = "PASS" if ok else "FAIL"
        extra = f" 互串:{crossed}" if crossed else ""
        print(f"[{mark}] #{s['id']}({s['type']}) {s['query']} → {paths}{extra}")
        if not ok:
            bad.append(s["id"])
    freight_bad = [b for b in bad if 1 <= b <= 5]
    n = len(samples) - len(bad)
    print(f"\nhit-rate@{top_k} = {n}/{len(samples)}(线 ≥{PASS_MIN});运费组未全过: {freight_bad or '无'}")
    ok_all = n >= PASS_MIN and not freight_bad
    print("EVAL PASS" if ok_all else "EVAL FAIL")
    return 0 if ok_all else 1


async def _run(samples, top_k, debug) -> int:
    try:
        if debug:
            get_settings().rag_score_threshold = 0.0  # 仅评估进程内生效,不落盘
        return await _main(samples, top_k)
    finally:
        await dispose_engine()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--debug", action="store_true", help="阈值临时置 0,打印原始排序用于诊断")
    args = ap.parse_args()
    samples = json.loads(Path("evals/rag_retrieval_samples.json").read_text(encoding="utf-8"))
    init_engine(get_settings())
    sys.exit(asyncio.run(_run(samples, args.top_k, args.debug)))
