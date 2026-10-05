"""ch09 T12:端到端验收钉 2/3/4/5/6(spec「验收映射」,单元级内存 store 面)。

一个 dict store 同时喂飞轮(标准化→查重→归并入队)与审核面(列表/详情/通过/
驳回)+评估链,串通「缺口→队列→审核→知识库→趋势」整条数据流;LLM/writer 走
注入位(真模型「再问即对」面=T10 手测取证 chunk53 + README P 步演示脚本,
非回归硬基线标注)。CLI trend/report 用纯函数+tmp 文件钉,真两轮实录在 T11
(run5/6 已进 eval_runs 活库,README 演示命令即读这两行)。
"""
from datetime import datetime
from types import SimpleNamespace

import pytest
from app.api import routes as routes_mod
from app.db import crud
from app.jobs import eval_cycle as E
from app.main import app
from app.services import flywheel as F
from app.services import review as R

_T = datetime(2026, 10, 5, 2, 2, 2)


class _Store:
    def __init__(self):
        self.lcq = {}     # id -> row
        self.queue = {}   # id -> row
        self.chunks = {}  # id -> row
        self.next_q = 1
        self.next_c = 1


class _FlyCrud:
    """飞轮面假 crud:merge 按真语义走(命中累加/未命中新建,matched 写回)。"""

    def __init__(self, store):
        self.store = store

    async def get_lcq_row(self, session, row_id):
        return self.store.lcq.get(row_id)

    async def pending_review_candidates(self, session, *, cap):
        rows = [(r.id, r.normalized_question)
                for r in self.store.queue.values() if r.review_status == "待审"]
        return rows[:cap], len(rows)

    async def merge_lcq_into_queue(self, session, lcq_id, *, matched_id,
                                   normalized_question, suggested_answer):
        s = self.store
        if matched_id is not None:
            q = s.queue[matched_id]
            q.occurrence_count += 1
            s.lcq[lcq_id].matched_review_id = matched_id
            return matched_id
        rq = SimpleNamespace(id=s.next_q, normalized_question=normalized_question,
                             ai_suggested_answer=suggested_answer,
                             occurrence_count=1, review_status="待审",
                             approved_answer=None, created_at=_T, updated_at=_T)
        s.queue[rq.id] = rq
        s.next_q += 1
        s.lcq[lcq_id].matched_review_id = rq.id
        return rq.id


class _RevCrud(_FlyCrud):
    """审核面假 crud:路由三端点 + publish_approved 消费的四个读写位。"""

    async def list_review_queue(self, session, *, status=None):
        rows = [r for r in self.store.queue.values()
                if status is None or r.review_status == status]
        return sorted(rows, key=lambda r: (-r.occurrence_count, -r.id))

    async def get_review_row(self, session, rq_id):
        return self.store.queue.get(rq_id)

    async def lock_review_row(self, session, rq_id):
        return self.store.queue.get(rq_id)

    async def review_sources(self, session, rq_id):
        return [r for r in self.store.lcq.values() if r.matched_review_id == rq_id]

    async def find_chunk_by_qa(self, session, category, questions, answer):
        for c in self.store.chunks.values():
            if (c.category, c.questions, c.answer) == (category, questions, answer):
                return c
        return None

    async def add_chunk_drafts(self, session, drafts, *, commit=True):
        out = []
        for d in drafts:
            c = SimpleNamespace(id=self.store.next_c, category=d.category,
                                questions=d.questions, answer=d.answer,
                                section_path=d.section_path,
                                content_type=d.content_type,
                                is_key_clause=d.is_key_clause,
                                vectorize_status="pending")
            self.store.next_c += 1
            self.store.chunks[c.id] = c
            out.append(c)
        return out

    async def approve_review(self, session, rq_id, approved_answer, chunk_id):
        q = self.store.queue[rq_id]
        if q.review_status != "待审":
            raise crud.ReviewConflict(f"id={rq_id} 已非待审(并发处置)")
        q.review_status = "通过"
        q.approved_answer = approved_answer
        self.store.chunks[chunk_id].vectorize_status = "done"

    async def reject_review(self, session, rq_id):
        q = self.store.queue[rq_id]
        if q.review_status != "待审":
            raise crud.ReviewConflict(f"id={rq_id} 已非待审(并发处置)")
        q.review_status = "驳回"
        return q


class _Sess:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


@pytest.fixture
def store(monkeypatch):
    st = _Store()
    fc, rc = _FlyCrud(st), _RevCrud(st)

    async def _norm(raw, settings):
        return SimpleNamespace(normalized_question="冻干不爱吃能退吗",
                               suggested_answer="未开封可退。")

    async def _dedup(raw, norm, cands, settings):
        return None if not cands else cands[0][0]
    monkeypatch.setattr(F, "crud", fc)
    monkeypatch.setattr(F, "normalize_llm", _norm)
    monkeypatch.setattr(F, "dedup_llm", _dedup)
    monkeypatch.setattr(F, "get_session_factory", lambda: (lambda: _Sess()))
    monkeypatch.setattr(routes_mod, "crud", rc)
    monkeypatch.setattr(R, "crud", rc)
    app.dependency_overrides[routes_mod.dep_db_session] = lambda: object()
    yield st, rc
    app.dependency_overrides.clear()


def _add_lcq(st, row_id, raw, chunks):
    st.lcq[row_id] = SimpleNamespace(
        id=row_id, raw_question=raw, source="ch05_gate", reason="低置信",
        retrieved_chunks=chunks, matched_review_id=None, created_at=_T)


async def _process(row_id):
    await F.process_lcq_row(row_id)


# ---------- 验收2:知识库没有→兜底落池→队列出现→详情有原话/快照 ----------

async def test_acceptance2_gap_flows_to_queue_with_sources(client, store):
    st, _ = store
    _add_lcq(st, 1, "我家猫不吃冻干能退吗", [{"chunk_id": 9, "score": 0.11, "text": "冻干"}])
    _add_lcq(st, 2, "冻干猫咪不赏脸可以退货不", [{"chunk_id": 9, "score": 0.13, "text": "冻干"}])
    await _process(1)
    await _process(2)  # 同标准问=查重命中累加

    r = await client.get("/api/review_queue", params={"status": "待审"})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1 and body[0]["occurrence_count"] == 2, \
        "两条原话归并成一个缺口(不双行),出现次数=2"
    assert body[0]["normalized_question"] == "冻干不爱吃能退吗"
    assert body[0]["ai_suggested_answer"] == "未开封可退。"

    d = await client.get(f"/api/review_queue/{body[0]['id']}/detail")
    assert d.status_code == 200
    dd = d.json()
    assert {s["raw_question"] for s in dd["sources"]} == \
        {"我家猫不吃冻干能退吗", "冻干猫咪不赏脸可以退货不"}, "详情=归并原话清单"
    assert dd["sources"][0]["retrieved_chunks"][0]["chunk_id"] == 9, "当轮召回快照可查"


# ---------- 验收3:审核通过后同问可召回(机器面;真模型面见 README P 步) ----------

async def test_acceptance3_approve_writes_recallable_chunk(client, store):
    st, rc = store
    _add_lcq(st, 1, "我家猫不吃冻干能退吗", None)
    await _process(1)
    rq = next(iter(st.queue.values()))
    embedded = []

    async def writer(chunk_id, text, settings):
        embedded.append((chunk_id, text))
        st.chunks[chunk_id].vectorize_status = "done"  # 真 writer=upsert+flush,假面记文

    cid = await R.publish_approved(None, rq, "未开封支持7天退货。",
                                   settings=object(), writer=writer)
    assert rq.review_status == "通过" and rq.approved_answer == "未开封支持7天退货。"
    chunk = st.chunks[cid]
    assert (chunk.category, chunk.questions, chunk.answer) == \
        (R.CATEGORY, "冻干不爱吃能退吗", "未开封支持7天退货。") and \
        chunk.vectorize_status == "done", "过=chunk+done,同问检索可召回"
    assert embedded and "未开封支持7天退货。" in embedded[0][1], "向量文本=类目+问+答"

    # 再问即对(机内近似):同三元换个新待审行再走审核=指纹命中,KB 零重做
    # (同行已「通过」再 publish=并发/迟到写手,被 M3-I2 条件更新正当拦截)
    with pytest.raises(crud.ReviewConflict):
        await R.publish_approved(None, rq, "未开封支持7天退货。",
                                 settings=object(), writer=lambda *a: None)
    rq2 = SimpleNamespace(id=99, normalized_question="冻干不爱吃能退吗",
                          ai_suggested_answer=None, occurrence_count=1,
                          review_status="待审", approved_answer=None)
    st.queue[99] = rq2
    seen = []
    cid2 = await R.publish_approved(None, rq2, "未开封支持7天退货。",
                                    settings=object(),
                                    writer=lambda *a: seen.append(a))
    assert cid2 == cid and seen == [] and len(st.chunks) == 1, \
        "同问同答再批=复用既有 chunk,向量零重做且置态照走"


async def test_acceptance3_reject_leaves_kb_untouched(client, store):
    st, _ = store
    _add_lcq(st, 1, "能不能改地址", None)
    await _process(1)
    rq = next(iter(st.queue.values()))
    r = await client.patch(f"/api/review_queue/{rq.id}", json={"status": "驳回"})
    assert r.status_code == 200 and r.json()["review_status"] == "驳回"
    assert st.chunks == {}, "驳回=零知识库写入"


# ---------- 验收4:👎 链→落池→飞轮→队列 ----------

async def test_acceptance4_downvote_pools_spawns_and_queues(client, store, monkeypatch):
    st, rc = store
    anchor = SimpleNamespace(retrieval_snapshot=[{"chunk_id": 3, "score": 0.2, "text": "t"}])
    rc.find_feedback_anchor = lambda session, cid, seq: _ret((anchor, "退款咋还没到账"))
    rc.add_low_confidence_question = _add_pool_capture(st)
    spawned = []
    monkeypatch.setattr(F, "spawn_process", spawned.append)

    r = await client.post("/api/feedback",
                          json={"conversation_id": 1, "seq": 2, "vote": "down"})
    assert r.status_code == 200 and r.json() == {"pooled": True}
    assert spawned == [1], "👎 落池成功即自触发流水线(拍板 2A 三入口平权)"
    row = st.lcq[1]
    assert row.source == "user_feedback" and row.matched_review_id is None

    await _process(1)
    assert len(st.queue) == 1 and next(iter(st.queue.values())).review_status == "待审"


def _ret(v):
    async def _c():
        return v
    return _c()


def _add_pool_capture(st):
    box = {"id": 0}

    async def add(session, *, conversation_id, raw_question, source, reason,
                  retrieved_chunks):
        box["id"] += 1
        rid = box["id"]
        st.lcq[rid] = SimpleNamespace(
            id=rid, raw_question=raw_question, source=source, reason=reason,
            retrieved_chunks=retrieved_chunks, matched_review_id=None, created_at=_T)
        return rid
    return add


# ---------- 验收5:eval_runs ≥2 行 → trend 出表(两轮差值) ----------

def _run(i, trig, n, r3, faith):
    return SimpleNamespace(id=i, triggered_by=trig, dataset_size=n, created_at=_T,
                           metrics={"recall_at_3": r3, "recall_at_10": r3,
                                    "mrr_at_10": r3, "faithfulness": faith})


def test_acceptance5_trend_table_from_two_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "TREND_FILE", tmp_path / "trend.md")

    async def fake_load():
        return [_run(5, "手动", 10, 1.0, 0.5), _run(6, "定时", 10, 0.9, None)]
    monkeypatch.setattr(E, "load_runs", fake_load)
    assert E.main(["--trend"]) == 0
    text = (tmp_path / "trend.md").read_text(encoding="utf-8")
    assert "| 5 |" in text and "| 6 |" in text and "-0.100" in text, \
        "两行都在+第二轮环比差值出表(真库行=run5/6,T11 实录)"


# ---------- 验收6:report 文件成(意图组+ASCII 控制台) ----------

def test_acceptance6_report_file_written(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(E, "REPORT_FILE", tmp_path / "report.md")
    monkeypatch.setattr(E, "fetch_traces", lambda st, days: [
        {"tags": ["intent:物流"], "totalCost": 0.02, "latency": 2.5},
        {"tags": ["intent:退款退货"], "totalCost": 0.05, "latency": 4.0},
        {"tags": ["chat"], "totalCost": 0.9, "latency": 1.0}])  # 无 intent=不进表
    assert E.main(["--report", "--days", "3"]) == 0
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "退款退货" in text and "chat组" not in text and "闲聊" not in text, \
        "intent 组进表,无 intent 不进"
    assert "0.050000" in text and "4.000" in text
    assert capsys.readouterr().out.isascii(), "GBK 红线:控制台 ASCII"
