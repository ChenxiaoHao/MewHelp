"""ch04 四策略对比评估(spec §6.2):老师题库 A/B/C/E × {dense,bm25,hybrid,hybrid_rerank}
分桶出 Hit@3/10、Recall@3/10、MRR@10;D 桶跑 hybrid_rerank 观测 top1 分数做闸1校准表。

uv run python evals/run_strategy_eval.py [--limit N] [--buckets A_policy,...] [--arms dense,...] [--no-cache]
产物:evals/reports/ch04_strategy_report.md(重跑覆盖)。评估臂关两个在线阈值
(rag_score_threshold/retrieval_low_conf_threshold → 0.0):测排序质量,不测在线拒答。
无阻断线:数字如实出,劣化进失败样例分析(spec §0-9);--limit 是成本闸。

断流韧性(本机系统代理对云侧 1-2 分钟级断流的实况对策,零触碰指标定义,dev-notes T11 ① 申报):
- 逐题进度侧车 evals/cache/strategy_progress.json:崩溃续跑,不丢已完成样本;
- 向量缓存 evals/cache/embed_cache.json:同文本只打一次云,且四臂同向量=更严格的全链同输入;
- rewrite/rerank 瞬态降级先重试一次再归因降级,降级样例数如实写进报告头部。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.db.engine import dispose_engine, init_engine  # noqa: E402
from app.rag import milvus_store, retriever  # noqa: E402
from app.rag.query_understanding import UnderstandResult, understand_query  # noqa: E402
from evals.teacher_csv import eval_question, load_questions  # noqa: E402

CACHE_FILE = Path("evals/cache/rewrite_cache.json")
EMBED_CACHE_FILE = Path("evals/cache/embed_cache.json")
PROGRESS_FILE = Path("evals/cache/strategy_progress.json")
REPORT_FILE = Path("evals/reports/ch04_strategy_report.md")
ANSWER_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")
ALL_ARMS = ("dense", "bm25", "hybrid", "hybrid_rerank")
METRIC_KEYS = ("hit@3", "hit@10", "recall@3", "recall@10", "mrr@10")

TEACHER_NOTES = """### 老师材料出入说明(只读原则,不改数据)
1. D3「能不能开纸质发票邮寄」标为应拒答,但语料 billing-shipping《可开票类型》实际可答——该题会拉低「正确拒答率」,如实呈现。
2. A22/E9 标准要点写银卡 95 折,语料为银卡 9 折(金卡才 95 折)——按语料评估。
3. FAQ 未满 99 收 10 元 与 billing《运费与包邮》6 元为跨文档矛盾——两文档并存入库,引用哪块都算命中(期望章节按原子子串匹配)。"""


def _load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_cache() -> dict:
    return _load_json(CACHE_FILE, {})


def _save_cache(cache: dict) -> None:
    _save_json(CACHE_FILE, cache)


# ---- embed 韧性:进程内替换 retriever._embed(dev-notes T11 ①)——同文本一次云调用 + 磁盘缓存 + 断流退避 ----
_orig_embed = retriever._embed
_embed_vec_cache: dict = {}


async def _embed_patient(text: str, st) -> list[float]:
    vec = _embed_vec_cache.get(text)
    if vec is not None:
        return vec
    for delay in (0, 10, 20, 40, 80, 160):  # 等过 1-2 分钟级代理断流窗口
        if delay:
            print(f"[strategy-eval] embed backoff {delay}s", flush=True)
            await asyncio.sleep(delay)
        try:
            vec = await _orig_embed(text, st)
            break
        except Exception:  # noqa: BLE001 — 云侧瞬态错退避重试;最终仍抛
            if delay == 160:
                raise
    _embed_vec_cache[text] = vec
    if len(_embed_vec_cache) % 10 == 0:  # 每 10 个新向量落一次盘:崩溃最多丢 10 次 embed
        _save_json(EMBED_CACHE_FILE, _embed_vec_cache)
    return vec


class _DegradeCounter(logging.Handler):
    """经 reranker 的降级日志计数(dev-notes T11 ①):不改判定,只数「本样本是否走了 RRF 降级序」。"""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.n = 0

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "rerank 不可用" in msg or "rerank 再次失败" in msg:
            self.n += 1


_degrade_log = _DegradeCounter()


async def _understand(query: str, st, cache: dict, counters: dict) -> UnderstandResult:
    hit = cache.get(query)
    if hit is not None:
        r = UnderstandResult(**hit)
        if not r.degraded:
            return r
        r2 = await understand_query(query, st)  # 瞬态曾把降级写进缓存:重跑时给一次修复机会
        if not r2.degraded:
            cache[query] = {"standard_query": r2.standard_query, "synonyms": r2.synonyms, "degraded": r2.degraded}
        else:
            counters["rewrite_degraded"] += 1
        return r2
    r = await understand_query(query, st)
    if r.degraded:  # 一次瞬态重试再归因(controller 断流纪律;缓存始终存最终回放的那份)
        r2 = await understand_query(query, st)
        if not r2.degraded:
            r = r2
        else:
            counters["rewrite_degraded"] += 1
    cache[query] = {"standard_query": r.standard_query, "synonyms": r.synonyms, "degraded": r.degraded}
    return r


async def _retrieve_retry(label: str, query: str, arm: str, st_eval, u) -> "retriever.RetrieveResult":
    """非 embed 路(Milvus/MySQL/回查)瞬态错同样先重试再谈失败;3 次全挂才抛(fail-fast 保数据)。"""
    for attempt in (1, 2, 3):
        try:
            return await retriever.retrieve(query, strategy=arm, settings=st_eval, understood=u)
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"[strategy-eval] {label} arm={arm} attempt {attempt} failed, backoff {attempt * 10}s", flush=True)
            await asyncio.sleep(attempt * 10)


def _mean(xs) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _calibrate(d_scores: list[float], ans_scores: list[float]):
    cands = sorted({round(s, 3) for s in d_scores + ans_scores})
    if cands:
        cands = cands[:: max(1, len(cands) // 12)]
    table, best = [], (float("inf"), 0.3)
    for t in cands or [0.3]:
        false_refuse = _mean([1.0 if s < t else 0.0 for s in ans_scores])   # 可答题被误拒
        over_conf = _mean([1.0 if s >= t else 0.0 for s in d_scores])       # D 题仍自信
        table.append((t, false_refuse, over_conf))
        if false_refuse + over_conf < best[0]:
            best = (false_refuse + over_conf, t)
    return table, round(best[1], 3)


async def _arm_record(q, arm, st_eval, u, counters) -> dict:
    """单臂一次测量;rerank 降级 → 重试一次;仍降级 → 如实记录并计数(spec §0-9)。"""
    for attempt in (1, 2):
        before = _degrade_log.n
        res = await _retrieve_retry(q.id, q.query, arm, st_eval, u)
        if arm != "hybrid_rerank" or _degrade_log.n == before:
            break  # 非精排臂,或本次精排成功
        if attempt == 1:
            print(f"[strategy-eval] {q.id} arm={arm} rerank 降级,重试一次", flush=True)
        else:
            counters["rerank_degraded"] += 1  # 重跑仍降级=正常结局,数字如实进表
    paths = [c.row.section_path for c in res.chunks]
    return {"m": eval_question(q.groups, paths, ks=(3, 10)), "top3": paths[:3],
            "top1": (res.chunks[0].score if res.chunks else None)}


async def run(args) -> int:
    st = get_settings()
    st_eval = st.model_copy(update={"rag_score_threshold": 0.0,
                                    "retrieval_low_conf_threshold": 0.0})
    init_engine(st)
    use_cache = not args.no_cache
    cache = _load_cache() if use_cache else {}
    if use_cache:  # 韧性件生效前先加载(仅 eval 进程内;--no-cache 全关,重测漂移语义不变)
        _embed_vec_cache.update(_load_json(EMBED_CACHE_FILE, {}))
        retriever._embed = _embed_patient
        _rlog = logging.getLogger("app.rag.reranker")
        _rlog.setLevel(logging.INFO)
        _rlog.addHandler(_degrade_log)
    counters = {"rewrite_degraded": 0, "rerank_degraded": 0}
    questions = load_questions()
    buckets = args.buckets.split(",") if args.buckets else list(ANSWER_BUCKETS)
    arms = args.arms.split(",") if args.arms else list(ALL_ARMS)
    assert set(arms) <= set(ALL_ARMS) and set(buckets) <= set(ANSWER_BUCKETS)
    answerable = [q for q in questions if q.bucket in buckets]
    if args.limit:
        answerable = answerable[: args.limit]
    d_questions = [q for q in questions if q.bucket == "D_absent"][: args.limit or 60]
    prog = _load_json(PROGRESS_FILE, {}) if use_cache else {}
    resumed = sum(1 for q in answerable if all(a in prog.get(q.id, {}) for a in arms))
    if resumed:
        print(f"[strategy-eval] 侧车续跑:{resumed} 题自进度复用", flush=True)

    rows: dict[tuple, list] = defaultdict(list)
    fails: list[str] = []
    ans_top1: dict[str, float] = {}
    started = datetime.now()
    for q in answerable:
        entry = prog.get(q.id, {})
        for arm in arms:
            if arm not in entry:
                u = await _understand(q.query, st, cache, counters)
                entry[arm] = await _arm_record(q, arm, st_eval, u, counters)
                prog[q.id] = entry
                if use_cache:
                    _save_json(PROGRESS_FILE, prog)
                    _save_cache(cache)
            rec = entry[arm]
            m = rec["m"]
            rows[(q.bucket, arm)].append(m)
            if arm == "hybrid_rerank" and rec["top1"] is not None:
                ans_top1[q.id] = rec["top1"]
            if m and m["hit@10"] == 0.0 and len(fails) < 40:
                fails.append(f"- `{q.id}` [{q.bucket}/{arm}] {q.query}\n  - 期望组: {q.groups}\n  - 实际 Top-3: {rec['top3']}")
        print(f"[strategy-eval] {q.id} ok", flush=True)
    if use_cache:
        _save_cache(cache)
        _save_json(EMBED_CACHE_FILE, _embed_vec_cache)
        _save_json(PROGRESS_FILE, prog)
    d_top1 = []
    for q in d_questions:
        u = await _understand(q.query, st, cache, counters)
        rec = await _arm_record(q, "hybrid_rerank", st_eval, u, counters)
        d_top1.append(rec["top1"] if rec["top1"] is not None else 0.0)
    if use_cache:
        _save_cache(cache)
        _save_json(EMBED_CACHE_FILE, _embed_vec_cache)

    client = milvus_store.get_client(st.milvus_uri)
    total_blocks = milvus_store.count_rows(client, st.milvus_collection)
    table, best_t = _calibrate(d_top1, list(ans_top1.values()))
    lines = [
        "# ch04 四策略对比评估报告",
        f"- 生成:{datetime.now():%Y-%m-%d %H:%M} · 题库:老师 `evals/run_rag.py`(300 题,只读) · 集合 `knowledge` 块数:{total_blocks} · 用时:{(datetime.now() - started).seconds // 60} 分",
        f"- 参数:arms={arms} buckets={buckets} limit={args.limit or '全量'} · rerank key:{'已配置' if st.rerank_api_key else '**未配置→hybrid_rerank 实为 RRF 降级序,D 桶校准数字不可用**'}" + (f" · 侧车续跑复用 {resumed} 题 · 降级样例:rewrite {counters['rewrite_degraded']} / rerank {counters['rerank_degraded']}(降级=正常路径,数字如实出)" if resumed or any(counters.values()) else ""),
        f"- 在线阈值在评估中关闭(两阈值→0.0),本表测**排序质量**;`rag_score_threshold` 本就只作用 dense 腿,四臂同输入。",
        "- **小库声明**:语料切块后仅约 " + str(total_blocks) + " 块,双腿 Top-50≈全库,Recall 天然偏高、**MRR 更有分辨力**(spec §6.2)。",
        "- **无阻断线声明**:未达观察目标不构成失败;劣化桶与失败样例见下,禁止为达标改数据/藏结果(spec §0-9,用户 2026-09-23 纠偏)。",
        "",
        "## 分桶 × 策略 指标均值",
        "",
        "| 桶 | 策略 | n | " + " | ".join(METRIC_KEYS) + " |",
        "|---|---|---|" + "---|" * len(METRIC_KEYS),
    ]
    for b in buckets:
        for a in arms:
            ms = [m for m in rows[(b, a)] if m]
            lines.append(f"| {b} | {a} | {len(ms)} | " +
                         " | ".join(f"{_mean([m[k] for m in ms]):.3f}" for k in METRIC_KEYS) + " |")
    lines += ["", "## 未命中样例(hit@10=0,最多 40 条)", ""] + (fails or ["(无)"])
    lines += ["", "## D 桶闸1阈值校准表", "",
              "top1 分数分布 + 误拒/误自信权衡(候选=实测 top1 分数分位;终值由 Task 12 回写 Settings 默认):", "",
              "| 阈值 | 可答题误拒率 | D题仍自信率 | 合计 |", "|---|---|---|---|"]
    lines += [f"| {t:.3f} | {f_:.3f} | {o:.3f} | {f_ + o:.3f} |" for t, f_, o in table]
    lines += [f"", f"**建议阈值:{best_t}**", "", TEACHER_NOTES, ""]
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"[strategy-eval] 报告 → {REPORT_FILE}")
    print(f"[strategy-eval] 降级统计 rewrite={counters['rewrite_degraded']} rerank={counters['rerank_degraded']}", flush=True)
    await dispose_engine()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ch04 四策略对比评估")
    parser.add_argument("--limit", type=int, default=0, help="题量成本闸(各桶截断)0=全量")
    parser.add_argument("--buckets", type=str, default="", help="逗号分隔桶名,默认 A/B/C/E 四答桶")
    parser.add_argument("--arms", type=str, default="", help="逗号分隔策略,默认四臂")
    parser.add_argument("--no-cache", action="store_true", help="禁 rewrite 缓存(重测改写漂移)")
    sys.exit(asyncio.run(run(parser.parse_args())))
