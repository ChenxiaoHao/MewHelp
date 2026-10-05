"""ch09 T11:评估流水线一轮一行——检索两指标(+可选 faith 抽样)+ trend + Langfuse 成本报表。

用法(三动作互斥,后两个不触评估轮):
  uv run python -m app.jobs.eval_cycle [--trigger 手动|定时] [--limit N] [--skip-faith]
  uv run python -m app.jobs.eval_cycle --trend
  uv run python -m app.jobs.eval_cycle --report [--days 7]
口径与 ch04 同律:检索臂 hybrid_rerank、在线双阈值→0.0(测排序质量不测在线拒答);
metrics 四键 {recall_at_3, recall_at_10, mrr_at_10, faithfulness} 进 eval_runs 一行,
dataset_size=实跑条数;faith 面复用 run_faith_eval 的 run_chain+judge 核(零改原脚本,
单样本失败重试外层不拦,降级数如实打表不进 metrics)。
GBK 红线:控制台只出 ASCII;中文表一律 UTF-8 落 evals/reports/。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.engine import dispose_engine, get_session_factory, init_engine  # noqa: E402
from app.db.models import EvalRuns  # noqa: E402

TREND_FILE = Path("evals/reports/ch09_eval_trend.md")
REPORT_FILE = Path("evals/reports/ch09_langfuse_report.md")
ANSWER_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")
FAITH_TIMEOUT_SECONDS = 240.0


def aggregate_metrics(ms: list[dict]) -> dict:
    """eval_question 键名 → eval_runs 四键;{}(D 桶/空组)不进母,空集定 0.0。"""
    def _mean(key: str) -> float:
        xs = [m[key] for m in ms if m and key in m]
        return round(sum(xs) / len(xs), 4) if xs else 0.0
    return {"recall_at_3": _mean("recall@3"), "recall_at_10": _mean("recall@10"),
            "mrr_at_10": _mean("mrr@10"), "faithfulness": None}


async def record_run(session, *, triggered_by: str, dataset_size: int,
                     metrics: dict) -> EvalRuns:
    row = EvalRuns(triggered_by=triggered_by, dataset_size=dataset_size,
                   metrics=metrics)
    session.add(row)
    await session.commit()
    return row


async def load_runs() -> list[EvalRuns]:
    async with get_session_factory()() as session:
        return list((await session.execute(
            select(EvalRuns).order_by(EvalRuns.id))).scalars().all())[-40:]


def _f3(v) -> str:
    return f"{v:.3f}" if v is not None else "—"


def _delta(cur, prev) -> str:
    if cur is None or prev is None:
        return "—"
    d = cur - prev
    return f"{d:+.3f}"


def trend_text(runs: list) -> str:
    """id 升序=时间正序(函数自排序,不仰仗查询序);环比差值对上一行,首轮=「基线」。"""
    runs = sorted(runs, key=lambda r: r.id)
    lines = ["# ch09 评估轮次趋势(eval_runs,按 id 升序=时间正序)", "",
             "| id | 触发 | n | 创建 | recall@3 环比 | recall@10 环比 | "
             "mrr@10 环比 | faith 环比 |",
             "|---|---|---|---|---|---|---|---|"]
    prev = None
    for r in runs:
        m = r.metrics or {}
        cur = (m.get("recall_at_3"), m.get("recall_at_10"), m.get("mrr_at_10"),
               m.get("faithfulness"))
        created = (r.created_at.strftime("%Y-%m-%d %H:%M")
                   if hasattr(r.created_at, "strftime") else str(r.created_at or ""))
        if prev is None:
            deltas = ["基线", "基线", "基线", "基线"]
        else:
            deltas = [_delta(c, p) for c, p in zip(cur, prev)]
        lines.append(f"| {r.id} | {r.triggered_by} | {r.dataset_size} | {created} | "
                     + " | ".join(f"{_f3(c)} ({d})" for c, d in zip(cur, deltas))
                     + " |")
        prev = cur
    lines.append("")
    lines.append(f"- 生成:{datetime.now():%Y-%m-%d %H:%M} · 共 {len(runs)} 轮")
    return "\n".join(lines)


def _intent_of(tags) -> str | None:
    for t in tags or []:
        if isinstance(t, str) and t.startswith("intent:"):
            return t[len("intent:"):]
    return None


def aggregate_traces(items: list[dict]) -> list[dict]:
    """Langfuse trace 列表按 intent:* tag 聚合 cost/latency;无 intent tag 不进表。"""
    acc: dict[str, dict] = {}
    for it in items:
        intent = _intent_of(it.get("tags"))
        if intent is None:
            continue
        a = acc.setdefault(intent, {"intent": intent, "traces": 0, "cost": 0.0,
                                    "latency_sum": 0.0})
        a["traces"] += 1
        a["cost"] += it.get("totalCost") or 0.0
        a["latency_sum"] += it.get("latency") or 0.0
    rows = []
    for a in acc.values():
        rows.append({"intent": a["intent"], "traces": a["traces"],
                     "cost": round(a["cost"], 6),
                     "avg_latency": round(a["latency_sum"] / a["traces"], 3)})
    rows.sort(key=lambda r: r["cost"], reverse=True)
    return rows


def fetch_traces(st, days: int) -> list[dict]:
    """近 days 天 trace 分页拉全。langfuse 未配置/不可达=明确报错(spec:不拖主流程)。"""
    if not (st.langfuse_host and st.langfuse_public_key and st.langfuse_secret_key):
        raise RuntimeError("langfuse 未配置(langfuse_host/public_key/secret_key 缺),"
                           "无法生成成本报表")
    items: list[dict] = []
    try:
        with httpx.Client(auth=(st.langfuse_public_key, st.langfuse_secret_key),
                          timeout=15, trust_env=False) as c:  # Windows 注册表代理不沾 localhost
            page = 1
            frm = (datetime.now(timezone.utc) - timedelta(days=days)
                   ).strftime("%Y-%m-%dT%H:%M:%SZ")
            while page <= 40:
                r = c.get(f"{st.langfuse_host.rstrip('/')}/api/public/traces",
                          params={"limit": 50, "page": page, "fromTimestamp": frm})
                r.raise_for_status()
                data = r.json().get("data") or []
                items.extend(data)
                if len(data) < 50:
                    break
                page += 1
    except Exception as exc:  # noqa: BLE001 —— 报表面单错,转成可读 RuntimeError
        raise RuntimeError(f"langfuse 不可达:{exc}") from exc
    return items


def report_text(rows: list[dict], days: int) -> str:
    lines = [f"# ch09 Langfuse 意图成本报表(近 {days} 天,tag=intent:*)", "",
             "| 意图 | trace 数 | 总成本(USD) | 平均延迟(s) |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['intent']} | {r['traces']} | {r['cost']:.6f} "
                     f"| {r['avg_latency']:.3f} |")
    lines += ["", f"- 生成:{datetime.now():%Y-%m-%d %H:%M} · "
              f"意图组数:{len(rows)}", ""]
    return "\n".join(lines)


async def run_cycle(args, st) -> tuple[dict, int]:
    """检索四指标轮(+未 skip 时 faith 抽样)。返回 (metrics, 实跑条数)。"""
    from app.rag import retriever
    from evals.teacher_csv import eval_question, load_questions

    st_eval = st.model_copy(update={"rag_score_threshold": 0.0,
                                    "retrieval_low_conf_threshold": 0.0})
    qs = [q for q in load_questions() if q.bucket in ANSWER_BUCKETS]
    if args.limit:
        qs = qs[: args.limit]
    ms = []
    for q in qs:
        res = await retriever.retrieve(q.query, strategy="hybrid_rerank",
                                       settings=st_eval)
        paths = [c.row.section_path for c in res.chunks]
        ms.append(eval_question(q.groups, paths, ks=(3, 10)))
        print(f"[eval-cycle] retrieve {q.id} ok", flush=True)
    metrics = aggregate_metrics(ms)
    if not args.skip_faith:
        from evals.run_faith_eval import judge, run_chain

        faithful = judged = degraded = 0
        for q in qs:
            try:
                answer, cites = await asyncio.wait_for(
                    run_chain(q.query, st, None), timeout=FAITH_TIMEOUT_SECONDS)
                v = await judge(q.query, answer, cites, st)
                judged += 1
                faithful += 1 if v.verdict == "faithful" else 0
                print(f"[eval-cycle] faith {q.id} -> {v.verdict}", flush=True)
            except Exception:  # noqa: BLE001 —— 单样本降级不拦整轮(数字如实)
                degraded += 1
                print(f"[eval-cycle] faith {q.id} DEGRADED skipped", flush=True)
        metrics["faithfulness"] = round(faithful / judged, 4) if judged else None
        if degraded:
            print(f"[eval-cycle] faith degraded={degraded}/{len(qs)} "
                  f"(excluded from denominator)", flush=True)
    return metrics, len(qs)


async def _amain(args) -> int:
    st = get_settings()
    if args.report:
        rows = aggregate_traces(fetch_traces(st, args.days))
        REPORT_FILE.parent.mkdir(exist_ok=True)
        REPORT_FILE.write_text(report_text(rows, args.days), encoding="utf-8")
        print(f"[eval-cycle] report groups={len(rows)} -> {REPORT_FILE.as_posix()}")
        return 0
    init_engine(st)
    try:
        if args.trend:
            runs = await load_runs()
            TREND_FILE.parent.mkdir(exist_ok=True)
            TREND_FILE.write_text(trend_text(runs), encoding="utf-8")
            print(f"[eval-cycle] trend rows={len(runs)} -> {TREND_FILE.as_posix()}")
            return 0
        metrics, n = await run_cycle(args, st)
        async with get_session_factory()() as session:
            row = await record_run(session, triggered_by=args.trigger,
                                   dataset_size=n, metrics=metrics)
        print(f"[eval-cycle] run id={row.id} n={n} trigger={args.trigger!r} ascii-summary: "
              f"recall@3={metrics['recall_at_3']} recall@10={metrics['recall_at_10']} "
              f"mrr@10={metrics['mrr_at_10']} "
              f"faith={metrics['faithfulness'] if metrics['faithfulness'] is not None else 'skip'}")
        return 0
    finally:
        await dispose_engine()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eval_cycle",
                                description="ch09 eval cycle: metrics row + trend + report")
    p.add_argument("--trigger", choices=("手动", "定时"), default="手动")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-faith", action="store_true")
    p.add_argument("--trend", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--days", type=int, default=7)
    args = p.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
