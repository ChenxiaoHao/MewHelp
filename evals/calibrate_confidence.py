"""ch09 T4:evidence_confidence 300 题校准(spec 置信度闸升级节;定形定值唯一出处)。

跑法:uv run python evals/calibrate_confidence.py [--limit N] [--no-cache]
面定义:D_absent 60 题=应拦面(漏放=闸判过),其余 240 题=应放面(误拦=闸判拒)。
定线:漏放 ≤5% 且误拦 ≤6.2%(ch05 基线 4.2%+2pp);冲突优先误拦不恶化。
证据分域:hybrid_rerank 精排分(RRF 降级样本剔除计数,不进标定域——运行期走旁路)。
embed/rewrite 缓存复用 run_strategy_eval 同文件;逐题分数侧车
evals/cache/confidence_scores.json(崩溃续跑;扫参只在侧车上做纯算术,
rerank 云调用每题至多一次)。报告 UTF-8 落 evals/reports/,控制台只 ASCII(GBK 红线)。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings                      # noqa: E402
from app.db.engine import dispose_engine, init_engine         # noqa: E402
from app.rag import confidence, retriever                     # noqa: E402
from evals import run_strategy_eval as rse                    # noqa: E402
from evals.teacher_csv import load_questions                  # noqa: E402

SCORES_FILE = Path("evals/cache/confidence_scores.json")
REPORT_FILE = Path("evals/reports/ch09_confidence_calibration.md")
LEAK_TARGET, FALSE_TARGET, FALSE_BASELINE = 0.05, 0.062, 0.042


def _load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


async def _collect(st, use_cache: bool, limit: int) -> dict:
    """每题实跑一次 hybrid_rerank,分数字段侧车;已缓存题直接跳过(侧车即进度)。"""
    st_eval = st.model_copy(update={"rag_score_threshold": 0.0,
                                    "retrieval_low_conf_threshold": 0.0})
    init_engine(st)
    cache = rse._load_cache() if use_cache else {}
    if use_cache:
        rse._embed_vec_cache.update(rse._load_json(rse.EMBED_CACHE_FILE, {}))
        retriever._embed = rse._embed_patient
        _rlog = logging.getLogger("app.rag.reranker")
        _rlog.setLevel(logging.INFO)
        _rlog.addHandler(rse._degrade_log)
    counters = {"rewrite_degraded": 0, "rerank_degraded": 0}
    side = rse._load_json(SCORES_FILE, {}) if use_cache else {}
    questions = load_questions()
    answerable = [q for q in questions if q.bucket in rse.ANSWER_BUCKETS]
    d_questions = [q for q in questions if q.bucket == "D_absent"]
    if limit:
        answerable, d_questions = answerable[:limit], d_questions[:limit]
    todo = [q for q in answerable + d_questions if q.id not in side]
    print(f"[calibrate] samples={len(answerable) + len(d_questions)} "
          f"resume={len(answerable) + len(d_questions) - len(todo)} todo={len(todo)}",
          flush=True)
    try:
        for q in todo:
            u = await rse._understand(q.query, st, cache, counters)
            degraded = False
            for attempt in (1, 2):  # _arm_record 同款:降级重试一次,仍降级如实记
                before = rse._degrade_log.n
                res = await rse._retrieve_retry(q.id, q.query, "hybrid_rerank", st_eval, u)
                if rse._degrade_log.n == before:
                    break
                degraded = (attempt == 2)
            side[q.id] = {"bucket": q.bucket,
                          "scores": [float(c.score) for c in res.chunks],
                          "degraded": degraded}
            if use_cache:
                _save_json(SCORES_FILE, side)
                rse._save_cache(cache)
            print(f"[calibrate] {q.id} ok n={len(side[q.id]['scores'])}"
                  f"{' DEGRADED' if degraded else ''}", flush=True)
        if use_cache:
            rse._save_json(rse.EMBED_CACHE_FILE, rse._embed_vec_cache)
            rse._save_cache(cache)
    finally:
        await dispose_engine()
    return side


def _evidence(scores: list[float]) -> list[dict]:
    return [{"chunk_id": i + 1, "score": s, "text": ""} for i, s in enumerate(scores)]


def _cfg(form: str, theta: float, floor: float, n_min: int, gap_min: float,
         w=(1.0, 0.0, 0.0)) -> SimpleNamespace:
    return SimpleNamespace(
        evidence_conf_form=form, evidence_conf_threshold=theta,
        evidence_conf_floor_eff=floor, evidence_conf_n_eff_min=n_min,
        evidence_conf_gap_min=gap_min, evidence_conf_w_top1=w[0],
        evidence_conf_w_n=w[1], evidence_conf_w_gap=w[2])


def _metrics(cfg, d_ev, a_ev) -> tuple[float, float]:
    leak = statistics.fmean(
        [1.0 if confidence.evaluate(ev, cfg).ok else 0.0 for ev in d_ev])
    false = statistics.fmean(
        [1.0 if not confidence.evaluate(ev, cfg).ok else 0.0 for ev in a_ev])
    return leak, false


def _theta_grid(all_scores: list[float]) -> list[float]:
    cands = sorted({round(s, 3) for s in all_scores if s > 0.0})
    if not cands:
        return [0.161]
    return cands[:: max(1, len(cands) // 16)]


def scan(side: dict) -> dict:
    d = [s for s in side.values() if s["bucket"] == "D_absent" and not s["degraded"]]
    a = [s for s in side.values() if s["bucket"] != "D_absent" and not s["degraded"]]
    d_ev, a_ev = [_evidence(x["scores"]) for x in d], [_evidence(x["scores"]) for x in a]
    all_top1 = [x["scores"][0] for x in d + a if x["scores"]]
    all_gap = [(x["scores"][0] - (x["scores"][1] if len(x["scores"]) > 1 else 0.0))
               for x in d + a if x["scores"]]
    thetas = _theta_grid(all_top1)
    best = None  # (sort_key, name, cfg, leak, false)
    # rule 形:θ × floor × n_min × gap_min(floor 只从 θ 域取代表值)
    floors = sorted({round(t, 3) for t in thetas})[:: max(1, len(thetas) // 6)] or [0.1]
    for floor in floors:
        for theta in thetas:
            for n_min in (1, 2, 3):
                for gap_min in (0.0, 0.05, 0.1, 0.2, 0.3):
                    cfg = _cfg("rule", theta, floor, n_min, gap_min)
                    leak, false = _metrics(cfg, d_ev, a_ev)
                    key = (false > FALSE_TARGET, false > FALSE_BASELINE,
                           round(leak, 4), round(false, 4), 0)
                    cand = (key, f"rule(θ={theta},floor={floor},n>={n_min},gap>={gap_min})",
                            cfg, leak, false)
                    if best is None or key < best[0]:
                        best = cand
    # sum 形:固定权重网格 × 合分阈(合分域 [0, max w_top1])
    for w in ((1, 0, 0), (.8, .1, .1), (.7, .2, .1), (.6, .2, .2),
              (.9, .05, .05), (.75, .15, .1), (.5, .3, .2)):
        for theta in _theta_grid([w[0] * t for t in all_top1]):
            cfg = _cfg("sum", theta, 0.1, 1, 0.0, w)
            leak, false = _metrics(cfg, d_ev, a_ev)
            key = (false > FALSE_TARGET, false > FALSE_BASELINE,
                   round(leak, 4), round(false, 4), 1)
            cand = (key, f"sum(w={w},θ={theta})", cfg, leak, false)
            if best is None or key < best[0]:
                best = cand
    n_degraded = sum(1 for s in side.values() if s["degraded"])
    return {"d_n": len(d), "a_n": len(a), "n_degraded": n_degraded,
            "gap_q": [round(q, 4) for q in statistics.quantiles(all_gap, n=4)],
            "top1_q": [round(q, 4) for q in statistics.quantiles(all_top1, n=4)],
            "best_name": best[1], "best_cfg": best[2],
            "leak": best[3], "false": best[4]}


def _report(res: dict, started: datetime) -> str:
    cfg = res["best_cfg"]
    lines = [
        "# ch09 evidence_confidence 300 题校准报告",
        f"- 生成:{datetime.now():%Y-%m-%d %H:%M} · 起点 {started:%H:%M} · "
        f"应拦面(D)={res['d_n']} · 应放面={res['a_n']} · 降级剔除={res['n_degraded']}",
        f"- 目标线:D 漏放 ≤{LEAK_TARGET:.1%} · 非 D 误拦 ≤{FALSE_TARGET:.1%}"
        f"(基线 {FALSE_BASELINE:.1%}+2pp),冲突优先误拦不恶化",
        f"- 分布:top1 四分位 {res['top1_q']} · gap 四分位 {res['gap_q']}",
        "",
        f"## 择优判定式:{res['best_name']}",
        "",
        "| 参数 | 值 |", "|---|---|",
        f"| form | {cfg.evidence_conf_form} |",
        f"| threshold(θ) | {cfg.evidence_conf_threshold} |",
        f"| floor_eff | {cfg.evidence_conf_floor_eff} |",
        f"| n_eff_min | {cfg.evidence_conf_n_eff_min} |",
        f"| gap_min | {cfg.evidence_conf_gap_min} |",
        f"| w_top1 / w_n / w_gap | {cfg.evidence_conf_w_top1} / "
        f"{cfg.evidence_conf_w_n} / {cfg.evidence_conf_w_gap} |",
        "",
        f"## 实测:D 桶漏放 {res['leak']:.3%} · 非 D 误拦 {res['false']:.3%}",
        "",
        "判定式=app/rag/confidence.py(与运行期同一核);分数侧车 "
        "`evals/cache/confidence_scores.json`;定值回填 `app/core/config.py` "
        "evidence_conf_* 键组。RRF 降级样本不在标定域(运行期走 RRF_DEGRADED_MAX 旁路)。",
    ]
    return "\n".join(lines) + "\n"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--scan-only", action="store_true",
                    help="只用侧车分数扫参(不触网;侧车缺失即报错)")
    args = ap.parse_args()
    started = datetime.now()
    if args.scan_only and not SCORES_FILE.exists():
        print("[calibrate] scan-only but sidecar missing", flush=True)
        return 2
    st = get_settings()
    side = await _collect(st, use_cache=not args.no_cache, limit=args.limit)
    res = scan(side)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(_report(res, started), encoding="utf-8", newline="\n")
    print(f"[calibrate] BEST {res['best_name']}", flush=True)
    print(f"[calibrate] leak={res['leak']:.4f} target={LEAK_TARGET} "
          f"false={res['false']:.4f} target={FALSE_TARGET} "
          f"degraded_excluded={res['n_degraded']}", flush=True)
    print(f"[calibrate] report -> {REPORT_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
