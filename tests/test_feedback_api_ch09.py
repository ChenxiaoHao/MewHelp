"""ch09 T6:POST /api/feedback(👎 后端化落池+当轮回捞;👍 204 不落)。

分支钉(brief Step1):up→204 零写入;down→反查第 seq 个**可见** assistant 行
(1 基;仅数 assistant 且 content 非 NULL——Review Focus 4 + M2-I1/M2-I2 改钉:
on_tool_calls 落库的 NULL-content 行不占号,与前端 assistantSeqFromHistory()
的 1 基计数及回载 `&& m.content` 过滤严格同律)→ raw_question=前紧邻 user →
快照回捞(NULL=空着)→ 落池 source=user_feedback/reason 记 seq → 200 {pooled:true};
会话缺/越界→404;重复👎允许重复落池。假 crud 同 ch05 流测套路;
find_feedback_anchor 真 SQL 走活库 integration 例。
"""

import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import delete

from app.core.config import get_settings
from app.db import crud
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.db.models import Conversation, Message


class _FakeCrud:
    def __init__(self, anchor=None, raw_question=None):
        self.calls = []
        self.spawns = []   # T7:👎 路落池成功即触发流水线的记录面
        self._anchor = anchor
        self._rq = raw_question

    async def find_feedback_anchor(self, session, conversation_id, seq):
        if self._anchor is None:
            return None, None
        return self._anchor, self._rq

    async def add_low_confidence_question(self, session, *, conversation_id,
                                          raw_question, source, reason,
                                          retrieved_chunks=None):
        self.calls.append({"cid": conversation_id, "q": raw_question,
                           "src": source, "reason": reason,
                           "snap": retrieved_chunks})
        return 77  # T7 起 crud 返回行 id,端点拿去触发


@pytest.fixture
def fake_routes_crud(monkeypatch):
    def _mk(fake):
        from app.api import routes as routes_mod
        from app.services import flywheel as flywheel_mod
        monkeypatch.setattr(routes_mod, "crud", fake)
        monkeypatch.setattr(flywheel_mod, "spawn_process",
                            lambda rid: fake.spawns.append(rid))
        from app.main import app
        app.dependency_overrides[routes_mod.dep_db_session] = lambda: object()
        return fake
    yield _mk
    from app.main import app
    app.dependency_overrides.clear()


async def test_up_vote_204_and_no_pool(client, fake_routes_crud):
    fake = fake_routes_crud(_FakeCrud())
    r = await client.post("/api/feedback",
                          json={"conversation_id": 7, "seq": 1, "vote": "up"})
    assert r.status_code == 204
    assert fake.calls == [] and fake.spawns == []


async def test_down_vote_pools_with_snapshot_and_reason(client, fake_routes_crud):
    anchor = SimpleNamespace(id=11, retrieval_snapshot=[
        {"chunk_id": 3, "score": 0.71, "text": "偏远地区条款"}])
    fake = fake_routes_crud(_FakeCrud(anchor=anchor, raw_question="免运费能抵附加费吗"))
    r = await client.post("/api/feedback",
                          json={"conversation_id": 7, "seq": 2, "vote": "down"})
    assert r.status_code == 200 and r.json() == {"pooled": True}
    assert fake.calls == [{"cid": 7, "q": "免运费能抵附加费吗",
                           "src": "user_feedback", "reason": "seq=2",
                           "snap": anchor.retrieval_snapshot}]
    assert fake.spawns == [77], "T7 拍板 2A:👎 落池成功即触发流水线(行 id 下传)"


async def test_down_snapshot_null_pools_empty_handed(client, fake_routes_crud):
    anchor = SimpleNamespace(id=12, retrieval_snapshot=None)  # 闲聊/旧行面
    fake = fake_routes_crud(_FakeCrud(anchor=anchor, raw_question="闲聊一句"))
    r = await client.post("/api/feedback",
                          json={"conversation_id": 7, "seq": 1, "vote": "down"})
    assert r.status_code == 200
    assert fake.calls[0]["snap"] is None, "回捞空着=快照 NULL 原样,不造假"


async def test_out_of_range_or_missing_conversation_404(client, fake_routes_crud):
    fake = fake_routes_crud(_FakeCrud(anchor=None))  # crud 反查未命中
    r = await client.post("/api/feedback",
                          json={"conversation_id": 999, "seq": 5, "vote": "down"})
    assert r.status_code == 404
    assert fake.calls == []


async def test_bad_body_422(client, fake_routes_crud):
    fake_routes_crud(_FakeCrud())
    r = await client.post("/api/feedback",
                          json={"conversation_id": 0, "seq": 1, "vote": "down"})
    assert r.status_code == 422, "conversation_id≥1/seq≥1(1 基)/vote 枚举由 schema 拦"
    r2 = await client.post("/api/feedback",
                           json={"conversation_id": 7, "seq": 0, "vote": "down"})
    assert r2.status_code == 422, "1 基同律:seq=0 是前端永远发不出的值,挡在 schema"
    r3 = await client.post("/api/feedback",
                           json={"conversation_id": 7, "seq": -1, "vote": "meh"})
    assert r3.status_code == 422


# ---- 真 SQL 面(integration,活库) ----


@pytest.fixture
async def session():
    init_engine(get_settings())
    try:
        async with get_session_factory()() as s:
            yield s
    finally:
        await dispose_engine()


@pytest.mark.integration
async def test_find_feedback_anchor_only_counts_visible_assistant(session):
    uid = f"it-ch09t6-{uuid.uuid4().hex[:10]}"
    conv = Conversation(user_id=uid)
    session.add(conv)
    await session.commit()
    rows = [
        ("user", "问题一"), ("assistant", "答一"),
        ("tool", "工具回显"), ("assistant", "答二(带快照)"),
    ]
    for role, content in rows:
        await crud.add_message(session, conv.id, role, content=content,
                               retrieval_snapshot=(
                                   [{"chunk_id": 9, "score": 0.5, "text": "t"}]
                                   if content.startswith("答二") else None))
    a1, q1 = await crud.find_feedback_anchor(session, conv.id, 1)  # M2-I2:1 基
    assert a1.content == "答一" and q1 == "问题一"
    a2, q2 = await crud.find_feedback_anchor(session, conv.id, 2)
    assert a2.content == "答二(带快照)" and a2.retrieval_snapshot[0]["chunk_id"] == 9
    assert q2 == "问题一", "前紧邻 user 行(tool 不算数)"
    miss, mq = await crud.find_feedback_anchor(session, conv.id, 3)
    assert miss is None and mq is None
    gone, _ = await crud.find_feedback_anchor(session, 999999, 1)
    assert gone is None
    await session.execute(delete(Message).where(Message.conversation_id == conv.id))
    await session.execute(delete(Conversation).where(Conversation.id == conv.id))
    await session.commit()


@pytest.mark.integration
async def test_anchor_skips_null_content_toolcall_rows(session):
    """M2-I1:工具轮中间 assistant 行(on_tool_calls 无前言文本→content=NULL)不占号。

    前端口径两证:live 流只 push 终答(index.html:601);回载 filter
    `&& m.content` 丢 NULL 行(index.html:717)。反查计数必须同律,否则走过
    工具的轮整体错位——👎 落池锚到 NULL 行,快照回捞恒空还回 200(静默错归因)。"""
    uid = f"it-ch09t6b-{uuid.uuid4().hex[:10]}"
    conv = Conversation(user_id=uid)
    session.add(conv)
    await session.commit()
    await crud.add_message(session, conv.id, "user", content="帮我查订单")
    await crud.add_message(session, conv.id, "assistant", content=None,
                           tool_calls=[{"id": "c1", "name": "query_order", "args": {}}])
    await crud.add_message(session, conv.id, "tool", content='{"order": 1}',
                           tool_call_id="c1")
    await crud.add_message(session, conv.id, "assistant", content="已发货喵",
                           retrieval_snapshot=[{"chunk_id": 8, "score": 0.6, "text": "s"}])
    anchor, q = await crud.find_feedback_anchor(session, conv.id, 1)
    assert anchor is not None and anchor.content == "已发货喵", \
        "NULL-content 行不得占号:第 1 个可见 assistant=终答行"
    assert anchor.retrieval_snapshot[0]["chunk_id"] == 8
    assert q == "帮我查订单"
    miss, _ = await crud.find_feedback_anchor(session, conv.id, 2)
    assert miss is None
    await session.execute(delete(Message).where(Message.conversation_id == conv.id))
    await session.execute(delete(Conversation).where(Conversation.id == conv.id))
    await session.commit()
