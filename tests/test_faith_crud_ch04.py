"""台账 upsert 复发流转(spec §3.2/附录A):created/updated/reactivated 三态;
复发自动退回未解决+清 resolution;resolved_at 保留;seen_count 递增;uk_eval_id 一题一行。"""

from datetime import datetime

import pytest

from app.db import crud
from app.db.models import FaithCase


class FakeResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class FakeSession:
    def __init__(self, existing=None, get_row=None):
        self.existing, self.added, self.commits, self._get = existing, [], 0, get_row

    async def execute(self, _stmt):
        return FakeResult(self.existing)

    async def get(self, _model, _cid):  # session.get 是协程,假件必须 async
        return self._get

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1


ARGS = dict(eval_id="A22", bucket="A_policy", query="银卡打折吗", answer="打了95折",
            reason="语料为9折", citations=[{"n": 1}], judge_model="m")


async def test_upsert_creates_when_absent():
    s = FakeSession(existing=None)
    out = await crud.upsert_faith_case(s, **ARGS)
    assert out == "created" and s.commits == 1
    row = s.added[0]
    assert row.status == "未解决" and row.seen_count == 1 and row.first_seen_at == row.last_seen_at


async def test_upsert_updates_unresolved_keeps_first_seen():
    old = FaithCase(eval_id="A22", bucket="A_policy", query="q", strategy="hybrid_rerank",
                    answer="a", reason="r", status="未解决", seen_count=3,
                    first_seen_at=datetime(2026, 1, 1), last_seen_at=datetime(2026, 1, 2))
    s = FakeSession(existing=old)
    out = await crud.upsert_faith_case(s, **ARGS)
    assert out == "updated" and old.seen_count == 4 and old.first_seen_at == datetime(2026, 1, 1)
    assert old.status == "未解决" and old.query == "银卡打折吗"


async def test_reactivation_resets_status_and_clears_resolution():
    old = FaithCase(eval_id="A22", bucket="A_policy", query="q", strategy="hybrid_rerank",
                    answer="a", reason="r", status="已解决", seen_count=2, resolution="老师标注有误",
                    first_seen_at=datetime(2026, 1, 1), last_seen_at=datetime(2026, 1, 2),
                    resolved_at=datetime(2026, 1, 3))
    s = FakeSession(existing=old)
    out = await crud.upsert_faith_case(s, **ARGS)
    assert out == "reactivated"
    assert old.status == "未解决" and old.resolution is None
    assert old.resolved_at == datetime(2026, 1, 3)  # 保留做复发显示(附录 A 注释语义)


async def test_rejudge_with_none_citations_overwrites_snapshot():
    """计划缺陷修订(实施者自判;锚点订正见 test_ledger_integration 同注)配套新案:spec §3.2/附录A「重判→更新 citations 快照」
    为无条件覆写——重判传 None 时旧快照被覆写为 NULL(非保留旧值);与集成层
    test_ledger_roundtrip_live 的 `mine.citations is None` 断言同向钉死(只增不减)。"""
    old = FaithCase(eval_id="A22", bucket="A_policy", query="q", strategy="hybrid_rerank",
                    answer="a", reason="r", status="未解决", seen_count=1,
                    citations=[{"n": 1}],
                    first_seen_at=datetime(2026, 1, 1), last_seen_at=datetime(2026, 1, 2))
    s = FakeSession(existing=old)
    out = await crud.upsert_faith_case(s, **{**ARGS, "citations": None})
    assert out == "updated" and old.citations is None


async def test_set_status_resolve_marks_and_unresolve_clears():
    row = FaithCase(eval_id="A22", bucket="A_policy", query="q", strategy="hybrid_rerank",
                    answer="a", reason="r", status="未解决", seen_count=1,
                    first_seen_at=datetime(2026, 1, 1), last_seen_at=datetime(2026, 1, 1))
    s = FakeSession(get_row=row)
    out = await crud.set_faith_case_status(s, 1, "已解决", "语料确实写9折")
    assert out.status == "已解决" and out.resolution == "语料确实写9折" and out.resolved_at is not None
    out2 = await crud.set_faith_case_status(s, 1, "未解决", None)
    assert out2.status == "未解决" and out2.resolution is None and out2.resolved_at == out.resolved_at
