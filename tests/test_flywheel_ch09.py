"""ch09 T7:飞轮流水线 process_lcq_row(标准化→查重→入队/归并)+触发+CLI 补扫。

mock LLM 面(brief Step1 六钉):命中累加不新建/未命中新建且 matched 指新行/
候选 ≤50 截断 WARN/LLM 崩行留 NULL 不外泄/Focus 6 同事务封口/补扫只吃 NULL 幂等。
真实 LLM+活库面在 T8(标注样例集)与 T12(e2e);本文件零网络零活库。
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from app.db import crud
from app.jobs import flywheel as flywheel_job
from app.services import flywheel as F
from app.services import refusals

LCQ = SimpleNamespace(id=5, raw_question="偏远地区能不能用免邮寄", matched_review_id=None)


class _Sess:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FC:
    """假 crud:状态按真 SQL 语义走——候选窗口截断、merge 后行离开 NULL 集。"""

    def __init__(self, row=LCQ, cands=(), total=None):
        self.row = row
        self.cands = list(cands)
        self.total = len(cands) if total is None else total
        self.merged = []
        self.null_ids = [5]

    async def get_lcq_row(self, session, row_id):
        return self.row if row_id == getattr(self.row, "id", None) else None

    async def pending_review_candidates(self, session, *, cap):
        return [(c.id, c.q) for c in self.cands[:cap]], self.total

    async def merge_lcq_into_queue(self, session, lcq_id, *, matched_id,
                                   normalized_question, suggested_answer):
        self.merged.append({"lcq": lcq_id, "matched": matched_id,
                            "nq": normalized_question, "sa": suggested_answer})
        self.null_ids.remove(lcq_id)
        return matched_id if matched_id is not None else 101

    async def unprocessed_lcq_ids(self, session, *, limit=None):
        ids = list(self.null_ids)
        return ids[:limit] if limit else ids


def _fake_llm(monkeypatch, norm_q="免邮寄权益能否抵扣偏远地区附加费",
              sa="可以,满99免首重。", matched=None, boom=None):
    calls = {"norm": [], "dedup": []}

    async def _norm(raw, settings):
        calls["norm"].append(raw)
        if boom == "norm":
            raise RuntimeError("llm down")
        return SimpleNamespace(normalized_question=norm_q, suggested_answer=sa)

    async def _dedup(raw, norm, cands, settings):
        calls["dedup"].append((raw, norm, list(cands)))
        if boom == "dedup":
            raise RuntimeError("dedup down")
        return matched

    monkeypatch.setattr(F, "normalize_llm", _norm)
    monkeypatch.setattr(F, "dedup_llm", _dedup)
    return calls


@pytest.fixture
def seams(monkeypatch):
    def _mk(fc):
        monkeypatch.setattr(F, "crud", fc)
        monkeypatch.setattr(F, "get_session_factory", lambda: (lambda: _Sess()))
        return fc
    return _mk


# ---- process_lcq_row 六钉 ----

async def test_hit_accumulates_no_new_row(seams, monkeypatch):
    fc = seams(_FC(cands=[SimpleNamespace(id=8, q="免运费能抵附加费吗")]))
    calls = _fake_llm(monkeypatch, matched=8)
    await F.process_lcq_row(5)
    assert fc.merged == [{"lcq": 5, "matched": 8,
                          "nq": "免邮寄权益能否抵扣偏远地区附加费",
                          "sa": "可以,满99免首重。"}], "命中=归并累加,不新建"


async def test_miss_creates_row_and_matched_points_new(seams, monkeypatch):
    fc = seams(_FC())
    _fake_llm(monkeypatch, matched=None)
    await F.process_lcq_row(5)
    assert fc.merged[0]["matched"] is None, "未命中=新建行(merge 契约)"


async def test_candidates_window_capped_and_warned(seams, monkeypatch, caplog):
    cands = [SimpleNamespace(id=i, q=f"q{i}") for i in range(1, 61)]  # 60>50
    fc = seams(_FC(cands=cands))
    calls = _fake_llm(monkeypatch, matched=None)
    with caplog.at_level(logging.WARNING):
        await F.process_lcq_row(5)
    seen = calls["dedup"][0][2]
    assert len(seen) == F.CANDIDATE_CAP == 50, "查重只见最新 50 窗口(按 updated_at 截)"
    assert any("50" in r.getMessage() for r in caplog.records), "截断必 WARN"


async def test_dedup_hallucinated_id_treated_as_miss(seams, monkeypatch, caplog):
    fc = seams(_FC(cands=[SimpleNamespace(id=8, q="甲")]))
    _fake_llm(monkeypatch, matched=999)  # 候选集外 id=幻觉
    with caplog.at_level(logging.WARNING):
        await F.process_lcq_row(5)
    assert fc.merged[0]["matched"] is None, "候选集外 matched 不得指向(防污染他行累加)"
    assert any("999" in r.getMessage() for r in caplog.records)


async def test_llm_crash_leaves_row_unprocessed_no_raise(seams, monkeypatch, caplog):
    fc = seams(_FC())
    _fake_llm(monkeypatch, boom="norm")
    with caplog.at_level(logging.WARNING):
        await F.process_lcq_row(5)          # 不外泄(spec:任务内异常全吞→WARN)
    assert fc.merged == [] and fc.null_ids == [5], "崩=行留 NULL 等补扫"


async def test_already_matched_or_missing_row_skipped(seams, monkeypatch):
    fc = seams(_FC(row=SimpleNamespace(id=5, raw_question="q", matched_review_id=8)))
    calls = _fake_llm(monkeypatch)
    await F.process_lcq_row(5)
    assert calls["norm"] == [], "matched 非 NULL=已处理,幂等边界不吃第二遍"
    await F.process_lcq_row(77)             # 行不存在
    assert fc.merged == []


# ---- crud 实现面:Focus 6 同事务封口(单 commit) ----

class _RecSession:
    def __init__(self):
        self.added, self.execs, self.commits, self.flushes = [], [], 0, 0

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        self.flushes += 1

    async def execute(self, stmt):
        self.execs.append(stmt)
        return SimpleNamespace(rowcount=1)  # 条件更新走正常路(guard 失败面=活库例钉)

    async def commit(self):
        self.commits += 1


async def test_merge_hit_single_commit():
    s = _RecSession()
    await crud.merge_lcq_into_queue(s, 5, matched_id=8,
                                   normalized_question="n", suggested_answer="a")
    assert s.commits == 1 and len(s.execs) == 2, "累加+matched 写回先全落,一次 commit 封口"


async def test_merge_new_row_single_commit():
    s = _RecSession()
    await crud.merge_lcq_into_queue(s, 5, matched_id=None,
                                   normalized_question="n", suggested_answer="a")
    assert s.commits == 1 and len(s.added) == 1 and s.flushes == 1 \
        and len(s.execs) == 1, "新行 flush 取 id 后 matched 同事务写回——半崩=双双不存在"


# ---- spawn 触发 ----

async def test_spawn_keeps_task_reference(seams, monkeypatch):
    fc = seams(_FC())
    done = []

    async def _rec(row_id):
        done.append(row_id)
    monkeypatch.setattr(F, "process_lcq_row", _rec)
    F.spawn_process(5)
    assert F._BG, "create_task 须存强引用——裸 task 可能被 GC 半路蒸发(Focus 6 前因)"
    await asyncio.gather(*F._BG)
    assert done == [5] and not F._BG, "done 回调清引用"


def test_spawn_without_loop_only_warns(caplog):
    with caplog.at_level(logging.WARNING):
        F.spawn_process(5)                  # 主线程无 loop:不抛(池写已成功,不该反噬)
    assert any("flywheel" in r.getMessage() for r in caplog.records)


async def test_refusals_pool_success_then_spawn(monkeypatch):
    seen = []

    async def fake_add(session, **kw):
        return 77
    monkeypatch.setattr(refusals.crud, "add_low_confidence_question", fake_add)
    monkeypatch.setattr(refusals, "get_session_factory",
                        lambda: (lambda: _Sess()))
    monkeypatch.setattr(F, "spawn_process", lambda rid: seen.append(rid))
    await refusals.pool_low_confidence(None, "q?", "ch05_gate", "r")
    assert seen == [77], "落池成功才触发(行 id 下传)"

    async def boom_add(session, **kw):
        raise RuntimeError("db down")
    monkeypatch.setattr(refusals.crud, "add_low_confidence_question", boom_add)
    seen.clear()
    await refusals.pool_low_confidence(None, "q?", "ch05_gate", "r")
    assert seen == [], "落池失败不触发"


# ---- CLI 补扫 ----

async def test_cli_dry_run_lists_without_processing(monkeypatch, capsys):
    fc = _FC(cands=())
    fc.null_ids = [5, 6]
    processed = []

    async def _rec(row_id):
        processed.append(row_id)
    monkeypatch.setattr(flywheel_job, "crud", fc)
    monkeypatch.setattr(flywheel_job, "get_session_factory",
                        lambda: (lambda: _Sess()))
    monkeypatch.setattr(F, "process_lcq_row", _rec)
    await flywheel_job._run(SimpleNamespace(limit=None, dry_run=True))
    out = capsys.readouterr().out
    assert processed == [], "dry-run 不动 LLM 不写库"
    assert "2" in out and "5" in out and "6" in out
    assert out.isascii(), "GBK 红线:控制台只出 ASCII"


async def test_cli_sweep_only_null_rows_idempotent(monkeypatch):
    fc = _FC()                              # null_ids=[5],merge 后离集(真 SQL 语义桩)
    monkeypatch.setattr(flywheel_job, "crud", fc)
    monkeypatch.setattr(flywheel_job, "get_session_factory",
                        lambda: (lambda: _Sess()))
    monkeypatch.setattr(F, "crud", fc)
    monkeypatch.setattr(F, "get_session_factory",
                        lambda: (lambda: _Sess()))
    _fake_llm(monkeypatch, matched=None)
    await flywheel_job._run(SimpleNamespace(limit=None, dry_run=False))
    assert fc.merged and fc.null_ids == []
    fc.merged.clear()
    await flywheel_job._run(SimpleNamespace(limit=None, dry_run=False))
    assert fc.merged == [], "二扫=零行:matched 写回即出 NULL 集,幂等"


# ---- 活库面(integration):真 SQL 队列形+累加+幂等(LLM 走注入位免网) ----

import uuid  # noqa: E402  (seam 段置前,活库例尾置)
from sqlalchemy import delete, select  # noqa: E402

from app.db.engine import dispose_engine, get_session_factory, init_engine  # noqa: E402
from app.db.models import Conversation, LowConfidenceQuestion, ReviewQueue  # noqa: E402
from app.core.config import get_settings  # noqa: E402


@pytest.fixture
async def session():
    init_engine(get_settings())
    try:
        async with get_session_factory()() as s:
            yield s
    finally:
        await dispose_engine()


@pytest.mark.integration
async def test_pipeline_live_queue_create_accumulate_idempotent(session):
    norm_q = f"it-ch09t7 免邮寄权益能否抵扣偏远地区附加费 {uuid.uuid4().hex[:8]}"

    async def _norm(raw, settings):
        return SimpleNamespace(normalized_question=norm_q, suggested_answer="可以。")

    async def _dedup(raw, norm, cands, settings):
        # 内容命中(非盲取窗首)——活库队列有存量待审行时假设不成立,M3 后修
        return next((cid for cid, q in cands if q == norm_q), None)

    row1 = await crud.add_low_confidence_question(
        session, conversation_id=None, raw_question="甲说法", source="ch05_gate",
        reason="r1", retrieved_chunks=None)
    await F.process_lcq_row(row1, normalizer=_norm, deduper=_dedup)
    qrows = (await session.execute(
        select(ReviewQueue).where(ReviewQueue.normalized_question == norm_q))
    ).scalars().all()
    assert len(qrows) == 1 and qrows[0].occurrence_count == 1 \
        and qrows[0].review_status == "待审", "未命中=新建待审行(count 1)"
    r1 = await session.get(LowConfidenceQuestion, row1)
    assert r1.matched_review_id == qrows[0].id, "matched 同事务指新行(Focus 6 封口)"

    row2 = await crud.add_low_confidence_question(
        session, conversation_id=None, raw_question="乙说法", source="user_feedback",
        reason="seq=1", retrieved_chunks=None)
    await F.process_lcq_row(row2, normalizer=_norm, deduper=_dedup)
    await session.refresh(qrows[0])
    assert qrows[0].occurrence_count == 2, "命中=累加不新建行"
    r2 = await session.get(LowConfidenceQuestion, row2)
    assert r2.matched_review_id == qrows[0].id

    await F.process_lcq_row(row1, normalizer=_norm, deduper=_dedup)  # 幂等重跑
    await session.refresh(qrows[0])
    assert qrows[0].occurrence_count == 2, "matched 非 NULL=出集,重跑不吃第二遍"
    await session.execute(delete(LowConfidenceQuestion)
                          .where(LowConfidenceQuestion.id.in_([row1, row2])))
    await session.execute(delete(ReviewQueue)
                          .where(ReviewQueue.id == qrows[0].id))
    await session.execute(delete(Conversation).where(Conversation.user_id.like("it-ch09t7%")))
    await session.commit()


@pytest.mark.integration
async def test_merge_concurrent_loser_noop_live(session):
    """M3-I1:entry 检查后输家才走到 merge(CLI 补扫×在线触发并发)——
    matched 写回条件更新挡双并:输家 no-op 回滚,不留孤儿队列行。"""
    nq = f"it-ch09t7m 并发输家不双并 {uuid.uuid4().hex[:8]}"
    row = await crud.add_low_confidence_question(
        session, conversation_id=None, raw_question="原话", source="ch05_gate",
        reason="r", retrieved_chunks=None)
    await session.commit()
    rq_a = await crud.merge_lcq_into_queue(session, row, matched_id=None,
                                           normalized_question=nq, suggested_answer="答")
    rq_b = await crud.merge_lcq_into_queue(session, row, matched_id=None,
                                           normalized_question=nq, suggested_answer="答")
    assert rq_b is None, "matched 已被赢家回填=本次 merge 整笔作废"
    got = (await session.execute(
        select(ReviewQueue).where(ReviewQueue.normalized_question == nq))).scalars().all()
    assert [r.id for r in got] == [rq_a], "不双并:表里只有赢家的行"
    lcq = await session.get(LowConfidenceQuestion, row)
    assert lcq.matched_review_id == rq_a and got[0].occurrence_count == 1
    await session.execute(delete(LowConfidenceQuestion)
                          .where(LowConfidenceQuestion.id == row))
    await session.execute(delete(ReviewQueue).where(ReviewQueue.id == rq_a))
    await session.commit()
