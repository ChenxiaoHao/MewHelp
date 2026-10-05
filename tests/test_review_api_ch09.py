"""ch09 T9:审核 API 三路由 + 核准回写 publish_approved(brief Step1 全覆盖)。

流转表:仅 待审→通过|驳回;通过必带 approved_answer(缺→422);其余流转 422;
行不存在/详情缺 404。Focus 5 半成功:publish 双落(chunk+向量)任一失败→502 且
状态留待审(置态动作只在 approve_review 内,崩在其前=状态不可能动)。
路由面假 crud+假 publish;服务面真逻辑假 crud 记录器+注入 writer;
指纹幂等(同问同答不双写 chunk)走活库 integration 例。
"""

from datetime import datetime
from types import SimpleNamespace

_T = datetime(2026, 10, 5, 1, 1, 1)

import pytest
from sqlalchemy import delete, select

from app.core.config import get_settings
from app.db import crud
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.db.models import KnowledgeChunk, ReviewQueue


def _row(**kw):
    base = dict(id=1, normalized_question="免邮寄能否抵附加费", ai_suggested_answer="可以",
                occurrence_count=3, review_status="待审", approved_answer=None,
                created_at=_T, updated_at=_T)
    base.update(kw)
    return SimpleNamespace(**base)


# ---------- 路由面(假 crud / 假 publish) ----------


class _RC:
    def __init__(self, rows=(), row=None, sources=()):
        self.rows, self.row, self.sources = list(rows), row, list(sources)
        self.list_calls, self.rejects, self.approves = [], [], []

    async def list_review_queue(self, session, *, status=None):
        self.list_calls.append(status)
        return self.rows

    async def get_review_row(self, session, rq_id):
        return self.row if self.row and self.row.id == rq_id else None

    async def review_sources(self, session, rq_id):
        return self.sources

    async def reject_review(self, session, rq_id):
        self.rejects.append(rq_id)
        return _row(id=rq_id, review_status="驳回")


@pytest.fixture
def fake_crud(monkeypatch):
    def _mk(rc, publisher=None):
        from app.api import routes as routes_mod
        from app.main import app
        monkeypatch.setattr(routes_mod, "crud", rc)
        if publisher is not None:
            monkeypatch.setattr(routes_mod.review, "publish_approved", publisher)
        app.dependency_overrides[routes_mod.dep_db_session] = lambda: object()
        return rc
    yield _mk
    from app.main import app
    app.dependency_overrides.clear()


async def test_list_status_filter_and_order(client, fake_crud):
    rc = fake_crud(_RC(rows=[_row(id=2, occurrence_count=9), _row(id=1, occurrence_count=3)]))
    r = await client.get("/api/review_queue", params={"status": "待审"})
    assert r.status_code == 200
    assert rc.list_calls == ["待审"]
    body = r.json()
    assert [b["id"] for b in body] == [2, 1] \
        and body[0]["occurrence_count"] == 9 and body[0]["review_status"] == "待审", \
        "occurrence desc 由 crud 排序,路由透传过滤与顺序"


async def test_detail_assembles_row_and_sources(client, fake_crud):
    lcq = SimpleNamespace(id=7, raw_question="口语原话", source="ch05_gate",
                          reason="低置信", retrieved_chunks=[{"chunk_id": 3, "score": 0.2,
                                                              "text": "片段"}],
                          created_at=_T)
    fake_crud(_RC(row=_row(id=1), sources=[lcq]))
    r = await client.get("/api/review_queue/1/detail")
    assert r.status_code == 200
    body = r.json()
    assert body["normalized_question"] == "免邮寄能否抵附加费"
    assert body["sources"] == [{"id": 7, "raw_question": "口语原话",
                                "source": "ch05_gate", "reason": "低置信",
                                "retrieved_chunks": [{"chunk_id": 3, "score": 0.2,
                                                      "text": "片段"}],
                                "created_at": "2026-10-05T01:01:01"}], "详情=行+归并原话/快照列表"
    r404 = await client.get("/api/review_queue/99/detail")
    assert r404.status_code == 404


async def test_patch_reject_ok(client, fake_crud):
    rc = fake_crud(_RC(row=_row(id=1)))
    r = await client.patch("/api/review_queue/1", json={"status": "驳回"})
    assert r.status_code == 200 and r.json()["review_status"] == "驳回"
    assert rc.rejects == [1]


async def test_patch_approve_needs_answer(client, fake_crud):
    calls = []
    fake_crud(_RC(row=_row(id=1)), publisher=lambda *a, **k: calls.append(1))
    r = await client.patch("/api/review_queue/1", json={"status": "通过"})
    assert r.status_code == 422 and calls == [], "通过必带 approved_answer"
    r2 = await client.patch("/api/review_queue/1",
                            json={"status": "通过", "approved_answer": "  "})
    assert r2.status_code == 422 and calls == [], "空白文案视同缺失"


async def test_patch_only_from_pending(client, fake_crud):
    for st in ("通过", "驳回"):  # 已定态再流转=422(重复通过红线,Focus 5)
        fake_crud(_RC(row=_row(id=1, review_status=st)))
        r = await client.patch("/api/review_queue/1", json={"status": "驳回"})
        assert r.status_code == 422, f"{st}→驳回 非法"
    fake_crud(_RC(row=None))
    r404 = await client.patch("/api/review_queue/99", json={"status": "驳回"})
    assert r404.status_code == 404
    fake_crud(_RC(row=_row(id=1)))
    r5 = await client.patch("/api/review_queue/1", json={"status": "待定"})
    assert r5.status_code == 422, "流转目标只认 通过|驳回(Literal 拦)"


async def test_patch_approve_success_returns_chunk_id(client, fake_crud):
    async def pub(session, row, answer, **kw):
        assert row.id == 1 and answer == "满99免首重"
        return 99
    fake_crud(_RC(row=_row(id=1)), publisher=pub)
    r = await client.patch("/api/review_queue/1",
                           json={"status": "通过", "approved_answer": "满99免首重"})
    assert r.status_code == 200
    assert r.json() == {"id": 1, "review_status": "通过", "chunk_id": 99}


async def test_patch_publish_fail_502_state_untouched(client, fake_crud):
    rc = fake_crud(_RC(row=_row(id=1)), publisher=_boom_publish)
    r = await client.patch("/api/review_queue/1",
                           json={"status": "通过", "approved_answer": "答"})
    assert r.status_code == 502, "Focus 5:KB 写失败不得置通过"
    assert rc.rejects == [], "失败路径不得动状态"


async def _boom_publish(session, row, answer, **kw):
    raise RuntimeError("milvus down")


# ---------- 服务面:真 publish_approved 逻辑,假 crud + 注入 writer ----------

from app.services import review as R  # noqa: E402


class _SC:
    def __init__(self, existing=None):
        self.existing, self.added, self.approved = existing, [], []

    async def find_chunk_by_qa(self, session, category, questions, answer):
        return self.existing

    async def add_chunk_drafts(self, session, drafts, commit=True):
        rows = [SimpleNamespace(id=50 + i, category=d.category, questions=d.questions,
                                answer=d.answer, vectorize_status="pending")
                for i, d in enumerate(drafts)]
        self.added.append((drafts, commit))
        return rows

    async def approve_review(self, session, rq_id, approved_answer, chunk_id):
        self.approved.append((rq_id, approved_answer, chunk_id))


@pytest.fixture
def svc(monkeypatch):
    def _mk(sc):
        monkeypatch.setattr(R, "crud", sc)
        return sc
    return _mk


async def test_publish_new_creates_embeds_approves(svc):
    seen = []

    async def writer(chunk_id, text, st):
        seen.append((chunk_id, text))

    sc = svc(_SC())
    cid = await R.publish_approved(object(), _row(id=1), "答文",
                                  settings=object(), writer=writer)
    assert cid == 50
    assert sc.added and sc.added[0][1] is True, "pending 行先落(单独 commit)"
    assert sc.approved == [(1, "答文", 50)], "置态只在双落之后(approve_review)"
    cid0, text = seen[0]
    assert text == f"{R.CATEGORY}\n免邮寄能否抵附加费\n答文", "三格拼接与 embed 同源(§3.1 律)"


async def test_publish_existing_done_skips_everything(svc):
    sc = svc(_SC(existing=SimpleNamespace(id=61, vectorize_status="done")))
    seen = []

    async def writer(chunk_id, text, st):
        seen.append(chunk_id)

    cid = await R.publish_approved(object(), _row(id=1), "答文",
                                  settings=object(), writer=writer)
    assert cid == 61 and sc.added == [] and seen == [], \
        "指纹已 done=KB 面零重做(不双写 chunk/不再 embed)"
    assert sc.approved == [(1, "答文", 61)], "但队列置态照走——复用也是通过"


async def test_publish_existing_pending_reembeds(svc):
    sc = svc(_SC(existing=SimpleNamespace(
        id=62, vectorize_status="pending", category=R.CATEGORY,
        questions="免邮寄能否抵附加费", answer="答文")))
    seen = []

    async def writer(chunk_id, text, st):
        seen.append(chunk_id)

    cid = await R.publish_approved(object(), _row(id=1), "答文",
                                  settings=object(), writer=writer)
    assert cid == 62 and sc.added == [] and seen == [62] and sc.approved == [(1, "答文", 62)]


async def test_publish_writer_crash_no_state_set(svc):
    sc = svc(_SC())

    async def writer(chunk_id, text, st):
        raise RuntimeError("embed 502")

    with pytest.raises(RuntimeError):
        await R.publish_approved(object(), _row(id=1), "答文",
                                 settings=object(), writer=writer)
    assert sc.approved == [], "Focus 5:向量崩=绝不置通过"


# ---------- 活库面(integration):真 crud,状态机+指纹幂等(writer 注入免 Milvus) ----------


@pytest.fixture
async def session():
    init_engine(get_settings())
    try:
        async with get_session_factory()() as s:
            yield s
    finally:
        await dispose_engine()


@pytest.mark.integration
async def test_approve_reject_live_state_machine(session):
    q = "it-ch09t9 免邮寄权益能否抵扣偏远地区附加费"
    rq1 = ReviewQueue(normalized_question=q, ai_suggested_answer="甲",
                      occurrence_count=2, review_status="待审")
    rq2 = ReviewQueue(normalized_question=q + "-乙", ai_suggested_answer="乙",
                      occurrence_count=1, review_status="待审")
    session.add_all([rq1, rq2])
    await session.commit()
    seen = []

    async def writer(chunk_id, text, st):
        seen.append((chunk_id, text))

    cid = await R.publish_approved(session, rq1, "核准答案", settings=get_settings(),
                                   writer=writer)
    await session.refresh(rq1)
    chunk = await session.get(KnowledgeChunk, cid)
    assert rq1.review_status == "通过" and rq1.approved_answer == "核准答案"
    assert chunk.vectorize_status == "done" and chunk.category == R.CATEGORY, \
        "过=写成才置态:chunk+向量(writer 面)+done 回填同请求内"
    assert seen and seen[0][0] == cid

    rq3 = ReviewQueue(normalized_question=q, ai_suggested_answer="丙",
                      occurrence_count=1, review_status="待审")
    session.add(rq3)
    await session.commit()
    cid2 = await R.publish_approved(session, rq3, "核准答案", settings=get_settings(),
                                    writer=writer)
    assert cid2 == cid and len(seen) == 1, "同问同答第二过=指纹命中,KB 零重做"
    n_chunks = len((await session.execute(
        select(KnowledgeChunk.id).where(KnowledgeChunk.id == cid))).all())
    assert n_chunks == 1

    await session.execute(delete(ReviewQueue).where(
        ReviewQueue.id.in_([rq1.id, rq2.id, rq3.id])))
    await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.id == cid))
    await session.commit()
