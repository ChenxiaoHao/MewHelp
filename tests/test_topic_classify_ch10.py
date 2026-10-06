"""ch10 T9 码面钉:labels_from_probs 阈值/兜底、选池 SQL 只要未归类、批切分、insert/upsert 调用面。

真模型活库真写(--limit 3)是集成验证步骤(训练资产就绪后),不在本单测。
"""

import json

import pytest

from app.jobs.topic_classify import collect_unclassified, store_results
from app.services.topic_classifier import labels_from_probs

C = ["退换货", "物流", "其他"]


def test_labels_from_probs_threshold_order_and_fallback():
    assert labels_from_probs([0.9, 0.2, 0.1], C, 0.5) == ["退换货"]
    assert labels_from_probs([0.2, 0.9, 0.9], C, 0.5) == ["物流", "其他"]
    assert labels_from_probs([0.5, 0.0, 0.1], C, 0.5) == ["退换货"]  # >= 边界命中
    assert labels_from_probs([0.1, 0.2, 0.3], C, 0.5) == ["其他"]  # 零命中兜底


def test_topic_classifier_rejects_vocabulary_drift(tmp_path):
    from app.services.topic_classifier import TopicClassifier

    (tmp_path / "topic_config.json").write_text(
        json.dumps({"classes": ["修理工"], "threshold": 0.5}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="词表漂移"):
        TopicClassifier(tmp_path)


class _Res:
    def __init__(self, rows=None, scalar=None):
        self._rows, self._scalar = rows or [], scalar

    def all(self):
        return self._rows

    def scalar(self):
        return self._scalar


class FakeSession:
    """记录每条语句文本;execute 按队列吐预置结果。"""

    def __init__(self, results=None):
        self.sqls: list[str] = []
        self.results = list(results or [])
        self.committed = 0

    async def execute(self, stmt):
        self.sqls.append(str(stmt.compile()))
        return self.results.pop(0) if self.results else _Res()

    async def commit(self):
        self.committed += 1


@pytest.mark.asyncio
async def test_collect_sql_only_unclassified_unless_rerun():
    s = FakeSession(results=[_Res(rows=[(1, "q")])])
    out = await collect_unclassified(s, limit=5, rerun=False)
    assert out == [(1, "q")]
    sql = s.sqls[0]
    assert "topic_classifications.id IS NULL" in sql
    assert "LIMIT" in sql.upper()

    s2 = FakeSession(results=[_Res(rows=[])])
    await collect_unclassified(s2, limit=None, rerun=True)
    assert "IS NULL" not in s2.sqls[0]
    assert "LIMIT" not in s2.sqls[0].upper()


@pytest.mark.asyncio
async def test_store_results_insert_then_update_only_on_rerun():
    s = FakeSession(results=[_Res(scalar=None)])  # 不存在 → insert
    n = await store_results(s, [(7, ["物流"])], rerun=False)
    assert n == 1 and s.committed == 1
    assert any(i.upper().lstrip().startswith("INSERT") for i in s.sqls)

    s2 = FakeSession(results=[_Res(scalar=3)])  # 已存在 + 非 rerun → 不写
    await store_results(s2, [(7, ["物流"])], rerun=False)
    assert not any(i.upper().lstrip().startswith("UPDATE") for i in s2.sqls)

    s3 = FakeSession(results=[_Res(scalar=3)])  # 已存在 + rerun → update 刷新
    await store_results(s3, [(7, ["物流"])], rerun=True)
    upd = [i for i in s3.sqls if i.upper().lstrip().startswith("UPDATE")]
    assert len(upd) == 1 and "topic_classifications" in upd[0]


@pytest.mark.asyncio
async def test_job_feeds_cleaned_text_to_classifier(monkeypatch):
    # 终审 I-4:训练面全是 clean() 后文本,推理入参必须同口径(消 train/serve skew)
    import app.jobs.topic_classify as job_mod
    import app.services.topic_classifier as svc_mod
    from finetune.clean import clean

    raw = "我的 13800138000 查下物流，地址北京朝阳区某某路１号"
    seen: list[list[str]] = []

    class SpyClf:
        def __init__(self, *a, **k): pass
        def classify_batch(self, texts, batch_size=32):
            seen.append(list(texts))
            return [["物流"] for _ in texts]

    async def fake_collect(session, limit, rerun):
        return [(1, raw)]
    async def fake_store(session, pairs, rerun):
        return len(pairs)

    monkeypatch.setattr(job_mod, "collect_unclassified", fake_collect)
    monkeypatch.setattr(job_mod, "store_results", fake_store)
    monkeypatch.setattr(job_mod, "_trace", lambda *a: None)
    monkeypatch.setattr(svc_mod, "TopicClassifier", SpyClf)
    monkeypatch.setattr(job_mod, "init_engine", lambda st: None)

    async def fake_dispose(): pass
    monkeypatch.setattr(job_mod, "dispose_engine", fake_dispose)

    class _Ctx:
        async def __aenter__(self): return object()
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(job_mod, "get_session_factory", lambda: (lambda: _Ctx()))

    await job_mod._amain(_mkargs())
    assert seen[0] == [clean(raw)]


def _mkargs():
    import argparse
    ns = argparse.Namespace(limit=1, rerun=False, dry_run=False)
    return ns

