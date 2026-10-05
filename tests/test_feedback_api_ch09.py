"""ch09 T6:POST /api/feedback(👎 后端化落池+当轮回捞;👍 204 不落)。

分支钉(brief Step1):up→204 零写入;down→反查第 seq 个**有答的轮**(1 基;
终I1 改钉:计数按轮不按行——锚=轮内末条「可见」assistant 行(content 真值且
tool_calls 空;前言工具行与终答同气泡不占号,作废说明行由同轮终答覆盖,
空串行不算),与前端 assistantSeqFromHistory() 的行→段折叠计数严格同律)→
raw_question=开该轮的 user 行 →
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
async def test_find_feedback_anchor_numbers_answered_turns_1_based(session):
    """终I1 改钉:形状从「一轮双答行」改为真实双轮(轮=1 气泡),钉 1 基/越界/幽灵会话。"""
    uid = f"it-ch09t6-{uuid.uuid4().hex[:10]}"
    conv = Conversation(user_id=uid)
    session.add(conv)
    await session.commit()
    rows = [
        ("user", "问题一"), ("assistant", "答一"),
        ("user", "问题二"), ("assistant", None), ("tool", "工具回显"),
        ("assistant", "答二(带快照)"),
    ]
    for role, content in rows:
        await crud.add_message(session, conv.id, role, content=content,
                               tool_calls=([{"id": "c1", "name": "t", "args": {}}]
                                           if content is None and role == "assistant"
                                           else None),
                               retrieval_snapshot=(
                                   [{"chunk_id": 9, "score": 0.5, "text": "t"}]
                                   if content and content.startswith("答二") else None))
    a1, q1 = await crud.find_feedback_anchor(session, conv.id, 1)  # M2-I2:1 基
    assert a1.content == "答一" and q1 == "问题一"
    a2, q2 = await crud.find_feedback_anchor(session, conv.id, 2)
    assert a2.content == "答二(带快照)" and a2.retrieval_snapshot[0]["chunk_id"] == 9
    assert q2 == "问题二", "问=开该轮的 user 行(工具轮不另计)"
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


@pytest.mark.integration
async def test_anchor_preamble_row_does_not_occupy_slot(session):
    """终I1(a):带前言的工具轮=一个气泡(index.html:601 每轮至多 push 一条),
    前言+tool_calls 行 content 非 NULL 但不得占号——逐行数会顶歪后续所有号,
    👎 锚到前言行(永远无快照)、raw_question 错拿上一问,静默错归因进池。"""
    uid = f"it-ch09fin-a-{uuid.uuid4().hex[:10]}"
    conv = Conversation(user_id=uid)
    session.add(conv)
    await session.commit()
    await crud.add_message(session, conv.id, "user", content="帮我查订单")
    await crud.add_message(session, conv.id, "assistant", content="稍等喵,我去查~",
                           tool_calls=[{"id": "c1", "name": "query_order", "args": {}}])
    await crud.add_message(session, conv.id, "tool", content='{"order": 1}',
                           tool_call_id="c1")
    await crud.add_message(session, conv.id, "assistant", content="已发货喵",
                           retrieval_snapshot=[{"chunk_id": 8, "score": 0.6, "text": "s"}])
    await crud.add_message(session, conv.id, "user", content="退款到哪了")
    await crud.add_message(session, conv.id, "assistant", content="审核中喵")
    a1, q1 = await crud.find_feedback_anchor(session, conv.id, 1)
    assert a1.content == "已发货喵" and q1 == "帮我查订单", \
        "seq1=第 1 个有答的轮,锚=轮内末条可见行,前言行不占号"
    assert a1.retrieval_snapshot[0]["chunk_id"] == 8
    a2, q2 = await crud.find_feedback_anchor(session, conv.id, 2)
    assert a2.content == "审核中喵" and q2 == "退款到哪了"
    miss, _ = await crud.find_feedback_anchor(session, conv.id, 3)
    assert miss is None
    await session.execute(delete(Message).where(Message.conversation_id == conv.id))
    await session.execute(delete(Conversation).where(Conversation.id == conv.id))
    await session.commit()


@pytest.mark.integration
async def test_anchor_cancel_note_never_occupies_slot(session):
    """终I1(b):ch08 超时作废说明行(graph.py drain,live 从未 push)落在 drain
    轮内,锚=该轮终答不顶号;带前言的确认预览轮 live 确有气泡(`if (answer)`
    收到流过的前言),占号且锚回退前言行;无前言的预览轮(content NULL)不占号。"""
    uid = f"it-ch09fin-b-{uuid.uuid4().hex[:10]}"
    conv = Conversation(user_id=uid)
    session.add(conv)
    await session.commit()
    await crud.add_message(session, conv.id, "user", content="帮我改地址")
    await crud.add_message(session, conv.id, "assistant", content="先确认一下喵~",
                           tool_calls=[{"id": "c1", "name": "modify_address", "args": {}}])
    await crud.add_message(session, conv.id, "tool", content="interrupted",
                           tool_call_id="c1")
    await crud.add_message(session, conv.id, "user", content="算了查物流")
    await crud.add_message(session, conv.id, "assistant",
                           content="上一轮待确认操作已超时作废。")
    await crud.add_message(session, conv.id, "assistant", content="在途喵",
                           retrieval_snapshot=[{"chunk_id": 6, "score": 0.9, "text": "s"}])
    await crud.add_message(session, conv.id, "user", content="转人工")
    await crud.add_message(session, conv.id, "assistant", content=None,
                           tool_calls=[{"id": "c2", "name": "handoff", "args": {}}])
    a1, q1 = await crud.find_feedback_anchor(session, conv.id, 1)
    assert a1.content == "先确认一下喵~" and q1 == "帮我改地址", \
        "预览轮流过了前言=live 有气泡,占号且锚回退前言行(轮内无终答行)"
    a2, q2 = await crud.find_feedback_anchor(session, conv.id, 2)
    assert a2.content == "在途喵" and q2 == "算了查物流", \
        "作废说明行不得当锚:drain 轮锚=轮内终答行"
    assert a2.retrieval_snapshot[0]["chunk_id"] == 6
    miss, _ = await crud.find_feedback_anchor(session, conv.id, 3)
    assert miss is None, "无前言的纯工具预览轮(content NULL)不占号"
    await session.execute(delete(Message).where(Message.conversation_id == conv.id))
    await session.execute(delete(Conversation).where(Conversation.id == conv.id))
    await session.commit()


@pytest.mark.integration
async def test_anchor_empty_content_answer_skipped(session):
    """终I1(c):content='' 的终答行非 NULL(SQL isnot(None) 拦不住)但不构成
    气泡(live `if (answer)` 拦 push),Python 真值判断排除。"""
    uid = f"it-ch09fin-c-{uuid.uuid4().hex[:10]}"
    conv = Conversation(user_id=uid)
    session.add(conv)
    await session.commit()
    await crud.add_message(session, conv.id, "user", content="第一问")
    await crud.add_message(session, conv.id, "assistant", content="")
    await crud.add_message(session, conv.id, "user", content="第二问")
    await crud.add_message(session, conv.id, "assistant", content="真答喵")
    a1, q1 = await crud.find_feedback_anchor(session, conv.id, 1)
    assert a1.content == "真答喵" and q1 == "第二问", \
        "空串终答不构成气泡:第 1 号=第二问的轮"
    miss, _ = await crud.find_feedback_anchor(session, conv.id, 2)
    assert miss is None
    await session.execute(delete(Message).where(Message.conversation_id == conv.id))
    await session.execute(delete(Conversation).where(Conversation.id == conv.id))
    await session.commit()
