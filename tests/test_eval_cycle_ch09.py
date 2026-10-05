"""ch09 T11:eval_cycle——评估流水线纯函数面 + 行落库形状 + CLI 动作分流。

真模型/活库两轮跑批+trend 出表走 CLI 实录(dev-notes 留痕,行数据进库作 T12
验收面);本文件零网零活库(除尾例 integration:limit=2 --skip-faith 真轮)。
控制台 GBK 红线:CLI stdout 一律 ASCII,中文表进 UTF-8 文件——两路都有钉。
"""

from types import SimpleNamespace

import pytest
from app.jobs import eval_cycle as E


# ---- 纯函数:指标聚合 ----

def test_aggregate_metrics_maps_and_means():
    ms = [{"hit@3": 1.0, "hit@10": 1.0, "recall@3": 1.0, "recall@10": 1.0,
           "mrr@10": 0.5},
          {"hit@3": 0.0, "hit@10": 1.0, "recall@3": 0.5, "recall@10": 0.75,
           "mrr@10": 0.25},
          {}]  # D 桶空 dict 不进母
    out = E.aggregate_metrics(ms)
    assert out == {"recall_at_3": 0.75, "recall_at_10": 0.875, "mrr_at_10": 0.375,
                   "faithfulness": None}
    assert E.aggregate_metrics([]) == {"recall_at_3": 0.0, "recall_at_10": 0.0,
                                       "mrr_at_10": 0.0, "faithfulness": None}


# ---- 行落库形状 ----

class _Sess:
    def __init__(self):
        self.added, self.commits = [], 0

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1


async def test_record_run_row_shape():
    s = _Sess()
    await E.record_run(s, triggered_by="手动", dataset_size=12,
                       metrics={"recall_at_3": 0.9, "recall_at_10": 0.95,
                                "mrr_at_10": 0.8, "faithfulness": None})
    row = s.added[0]
    assert (row.triggered_by, row.dataset_size) == ("手动", 12)
    assert row.metrics["recall_at_3"] == 0.9 and "faithfulness" in row.metrics
    assert s.commits == 1, "一轮一行一次 commit"


# ---- trend 纯函数:升序 + 环比差值 ----

def _run(i, trig, n, r3, r10, mrr, faith):
    return SimpleNamespace(id=i, triggered_by=trig, dataset_size=n, created_at=None,
                           metrics={"recall_at_3": r3, "recall_at_10": r10,
                                    "mrr_at_10": mrr, "faithfulness": faith})


def test_trend_table_order_and_deltas():
    runs = [_run(2, "手动", 10, 0.80, 0.90, 0.70, None),
            _run(1, "定时", 10, 0.75, 0.85, 0.65, 0.9)]
    text = E.trend_text(runs)
    assert "0.750" in text and "0.800" in text, "id 升序=时间正序"
    assert "+0.050" in text, "环比差值(第二轮 recall@3 +0.05)"
    assert "—" in text, "faithfulness=None 占位横线"
    idx = text.index("0.750")
    assert idx < text.index("0.800")
    assert E.trend_text([]).startswith("#"), "空表不崩,仍出标题"
    assert E.trend_text([_run(1, "定时", 3, 0.5, 0.5, 0.5, 0.5)]).count("|") > 8


def test_trend_single_run_no_delta_column_value():
    text = E.trend_text([_run(1, "定时", 3, 0.5, 0.6, 0.7, 0.8)])
    assert "0.500" in text and "环比" in text  # 首轮差值列=基线标记而非数字


# ---- report 纯聚合:intent tag 分组 ----

def test_aggregate_traces_groups_by_intent():
    items = [
        {"tags": ["intent:物流", "chat"], "totalCost": 0.01, "latency": 2.0},
        {"tags": ["intent:物流"], "totalCost": None, "latency": 1.0},
        {"tags": ["intent:退款退货"], "totalCost": 0.05, "latency": 4.0},
        {"tags": ["chat"], "latency": 0.5},                # 无 intent=不进表
    ]
    rows = E.aggregate_traces(items)
    by = {r["intent"]: r for r in rows}
    assert set(by) == {"物流", "退款退货"}
    assert by["物流"]["traces"] == 2 and abs(by["物流"]["cost"] - 0.01) < 1e-9
    assert by["退款退货"]["avg_latency"] == 4.0
    assert rows[0]["intent"] == "退款退货", "按 cost 降序打表"


def test_report_fetch_missing_config_clear_error(monkeypatch):
    st = SimpleNamespace(langfuse_host="", langfuse_public_key="", langfuse_secret_key="")
    with pytest.raises(RuntimeError, match="langfuse"):
        E.fetch_traces(st, days=7)


def test_report_fetch_unreachable_clear_error(monkeypatch):
    import httpx

    st = SimpleNamespace(langfuse_host="http://127.0.0.1:1", langfuse_public_key="pk",
                         langfuse_secret_key="sk")

    def _boom(*a, **k):
        raise httpx.ConnectError("nope")
    monkeypatch.setattr(E.httpx, "Client", _boom)
    with pytest.raises(RuntimeError, match="langfuse"):
        E.fetch_traces(st, days=7)


# ---- CLI 动作分流:trend/report 不触评估轮 ----

def test_cli_action_flags_skip_cycle(monkeypatch, tmp_path, capsys):
    def _boom(*a, **k):
        raise AssertionError("cycle must not run in --trend/--report mode")
    monkeypatch.setattr(E, "run_cycle", _boom)
    monkeypatch.setattr(E, "TREND_FILE", tmp_path / "trend.md")
    monkeypatch.setattr(E, "REPORT_FILE", tmp_path / "report.md")

    async def fake_rows():
        return [_run(1, "定时", 3, 0.5, 0.6, 0.7, None)]
    monkeypatch.setattr(E, "load_runs", fake_rows)
    r_trend = E.main(["--trend"])
    monkeypatch.setattr(E, "fetch_traces", lambda st, days: [{"tags": ["intent:物流"],
                                                              "totalCost": 0.02, "latency": 1.0}])
    r_report = E.main(["--report"])
    assert (r_trend, r_report) == (0, 0)
    out = capsys.readouterr().out
    assert out.isascii(), "GBK 红线:CLI 摘要 ASCII,中文表进文件"
    assert (tmp_path / "trend.md").read_text(encoding="utf-8")
    assert "物流" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_cli_bad_trigger_rejected():
    with pytest.raises(SystemExit):
        E.main(["--trigger", "随缘"])


# ---- 终I2:评估轮必须携 settings=st_eval(阈值归零随参传,不靠在线默认) ----


async def test_run_cycle_threads_eval_settings(monkeypatch):
    """终I2:retrieve 不带 settings 时读在线 get_settings(),闸1
    (retrieval_low_conf_threshold)在跑批里照样拒→recall/mrr 被压低,
    与 ch04 基线及 evals/calibrate_confidence.py(携 st_eval)不可比。"""
    import app.rag.retriever as R
    import evals.teacher_csv as tc
    from app.core.config import get_settings

    calls = []

    async def fake_retrieve(query, *, strategy="hybrid_rerank", settings=None, **kw):
        calls.append(settings)
        return SimpleNamespace(chunks=[])
    monkeypatch.setattr(R, "retrieve", fake_retrieve)
    monkeypatch.setattr(tc, "load_questions", lambda *a, **k: [SimpleNamespace(
        id="q1", bucket="A_policy", query="冻干能退吗", groups=[["faq/退货"]])])
    args = SimpleNamespace(limit=1, skip_faith=True, trigger="手动")
    metrics, n = await E.run_cycle(args, get_settings())
    assert n == 1 and len(calls) == 1
    assert calls[0] is not None, "retrieve 必须显式收到 settings(不靠在线默认)"
    assert (calls[0].rag_score_threshold,
            calls[0].retrieval_low_conf_threshold) == (0.0, 0.0), "评估轮双闸归零"
    assert metrics["recall_at_3"] == 0.0, "零命中如实打表"


# ---- 活库最小轮(integration):limit=2 --skip-faith 真检索落一行再清掉 ----

@pytest.mark.integration
async def test_cycle_live_row_and_cleanup():
    from sqlalchemy import delete, select

    from app.core.config import get_settings
    from app.db.engine import dispose_engine, get_session_factory, init_engine
    from app.db.models import EvalRuns

    st = get_settings()
    init_engine(st)
    try:
        ns = SimpleNamespace(limit=2, skip_faith=True, trigger="手动")
        metrics, n = await E.run_cycle(ns, st)
        assert set(metrics) == {"recall_at_3", "recall_at_10", "mrr_at_10", "faithfulness"}
        assert metrics["faithfulness"] is None and n == 2
        async with get_session_factory()() as session:
            row = await E.record_run(session, triggered_by=ns.trigger,
                                     dataset_size=n, metrics=metrics)
            got = (await session.execute(
                select(EvalRuns).where(EvalRuns.id == row.id))).scalar_one()
            assert got.dataset_size == 2
            await session.execute(delete(EvalRuns).where(EvalRuns.id == row.id))
            await session.commit()
    finally:
        await dispose_engine()
