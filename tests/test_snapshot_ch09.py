"""ch09 T5:retrieval_snapshot 随 assistant 终答行落库(👎 回捞数据源)。

三形态(brief Step1):knowledge 轮终答行含快照 [{chunk_id,score,text截断}]、
闲聊轮 NULL、text 超 retrieval_snapshot_text_max 截断。写点=final assistant
行两处(react 自落 + graph 固定话术出口),同经 crud.add_message;
活库往返与 test_db_ch09_roundtrip 同款实 MySQL(integration)。
"""

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import delete

from app.core.config import Settings, get_settings
from app.db import crud
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.db.models import Conversation, Message
from app.rag import retriever as retriever_mod
from app.rag import confidence as confidence_mod
from app.rag.retriever import RetrieveResult, ScoredRow
from app.services.persistence import DBChatPersister
from app.workflows import graph as graph_mod
from app.workflows.graph import stream_graph_turn


def _st(**over):
    base = dict(openai_base_url="x", openai_api_key="x", model_name="m",
                retrieval_snapshot_text_max=8)
    return Settings(_env_file=None, **{**base, **over})


# ---- 快照形制(与 T4 落池快照同源一个 builder) ----

def test_evidence_snapshot_truncates_at_cap():
    text = "退换货政策" * 10
    snap = confidence_mod.evidence_snapshot(
        [{"chunk_id": 1, "score": 0.9, "text": text}], _st())
    assert snap == [{"chunk_id": 1, "score": 0.9, "text": text[:8]}]


def test_evidence_snapshot_empty_is_empty_list():
    assert confidence_mod.evidence_snapshot([], _st()) == []
    assert confidence_mod.evidence_snapshot(None, _st()) == []


# ---- persister 转发 ----

def test_persister_forwards_snapshot_kw(monkeypatch):
    seen = []

    async def fake_add(session, conversation_id, role, content=None,
                       tool_calls=None, tool_call_id=None,
                       retrieval_snapshot=None):
        seen.append({"role": role, "content": content,
                     "snap": retrieval_snapshot})
        return SimpleNamespace(id=1)

    async def _run():
        p = DBChatPersister(session=object(), conversation_id=3)
        snap = [{"chunk_id": 7, "score": 0.5, "text": "片段"}]
        await p.on_final_answer(3, "答案A", retrieval_snapshot=snap)
        await p.on_final_answer(3, "答案B")  # 老调用形:不传= NULL
    monkeypatch.setattr(crud, "add_message", fake_add)
    asyncio.run(_run())
    assert seen[0] == {"role": "assistant", "content": "答案A",
                       "snap": [{"chunk_id": 7, "score": 0.5, "text": "片段"}]}
    assert seen[1]["snap"] is None


# ---- 活库往返(add_message 新参) ----

@pytest.fixture
async def session():
    init_engine(get_settings())
    try:
        async with get_session_factory()() as s:
            yield s
    finally:
        await dispose_engine()


@pytest.mark.integration
async def test_add_message_snapshot_roundtrip_and_legacy_null(session):
    uid = f"it-ch09t5-{uuid.uuid4().hex[:10]}"
    conv = Conversation(user_id=uid)
    session.add(conv)
    await session.commit()
    snap = [{"chunk_id": 41, "score": 0.88, "text": "偏远地区附加费说明"}]
    m1 = await crud.add_message(session, conv.id, "assistant",
                                content="答", retrieval_snapshot=snap)
    m2 = await crud.add_message(session, conv.id, "assistant", content="闲聊答")
    await session.refresh(m1)
    await session.refresh(m2)
    assert m1.retrieval_snapshot == snap, "JSON 含中文逐字节往返"
    assert m2.retrieval_snapshot is None, "旧调用形=NULL(闲聊面)"
    await session.execute(delete(Message).where(Message.conversation_id == conv.id))
    await session.execute(delete(Conversation).where(Conversation.id == conv.id))
    await session.commit()


# ---- 全链两形态(stream_graph_turn + recorder persister) ----

class _FakeModel:
    def __init__(self, intent="商品咨询"):
        self.intent = intent

    def bind_tools(self, tools, **kw):
        return self

    async def ainvoke(self, msgs, **kw):
        return AIMessage(content=json.dumps(
            {"intent": self.intent, "confidence": 0.9}, ensure_ascii=False))

    async def astream(self, msgs, **kw):
        yield AIMessageChunk(content="政策是七天无理由。")


class _Rec:
    def __init__(self):
        self.final_calls = []

    async def on_final_answer(self, conversation_id, content,
                              retrieval_snapshot=None):
        self.final_calls.append({"content": content, "snap": retrieval_snapshot})

    async def on_tool_calls(self, *a, **kw):
        pass

    async def on_tool_result(self, *a, **kw):
        pass


@pytest.fixture(autouse=True)
def fresh_checkpointer():
    graph_mod.reset_checkpointer()
    yield
    graph_mod.reset_checkpointer()


def _patch_retrieve(monkeypatch, chunks):
    async def fake_retrieve(query, *, strategy="hybrid_rerank", category=None,
                            settings=None, understood=None):
        return RetrieveResult(chunks=chunks)
    monkeypatch.setattr(retriever_mod, "retrieve", fake_retrieve)


def test_knowledge_turn_final_call_carries_snapshot(monkeypatch):
    row = SimpleNamespace(category="退换货", questions="偏远地区运费",
                          answer="包邮权益可抵扣。" + "补" * 40)
    _patch_retrieve(monkeypatch, [ScoredRow(chunk_id=41, score=0.95, row=row)])
    st = _st()
    rec = _Rec()

    async def _run():
        msg = SimpleNamespace(role="user", content="偏远地区免运费吗")
        return [f async for f in stream_graph_turn(
            [msg], st, _FakeModel(), conversation_id=None, persister=rec)]

    frames = asyncio.run(_run())
    assert frames
    assert rec.final_calls, "终答行落库调用须发生"
    snap = rec.final_calls[-1]["snap"]
    assert snap and snap[0]["chunk_id"] == 41 and snap[0]["score"] == 0.95
    assert len(snap[0]["text"]) == 8, "vector_text 拼接体超 cap=8 → 截断"


# ---- react 消费面:M2-C-1 显式键契约 ----

def _react_state(**over):
    st = {"messages": [SimpleNamespace(content="q")], "conversation_id": 1,
          "evidence": [{"chunk_id": 1, "score": 0.9, "text": "片段"}]}
    return {**st, **over}


def _drive_react(state, persister):
    from app.agents.react import react_agent_stream

    class _M:
        def bind_tools(self, tools, **kw):
            return self

        async def astream(self, msgs, **kw):
            yield AIMessageChunk(content="答案喵。")

    async def _run():
        return [e async for e in react_agent_stream(
            state, _st(), _M(), persister=persister)]
    return asyncio.run(_run())


def test_react_explicit_none_key_wins_over_evidence():
    rec = _Rec()
    _drive_react(_react_state(retrieval_snapshot=None), rec)
    assert rec.final_calls[-1]["snap"] is None, \
        "agent_node 显式交 None(五段保险丝形状)不得回退 evidence 现算"


def test_react_legacy_fallback_computes_from_evidence():
    rec = _Rec()
    _drive_react(_react_state(), rec)
    assert rec.final_calls[-1]["snap"] == [
        {"chunk_id": 1, "score": 0.9, "text": "片段"}], \
        "无显式键=旧直调面契约:从 evidence 现算(ch05/06 测形不变)"


def test_chitchat_turn_snapshot_none(monkeypatch):
    async def boom_retrieve(*a, **kw):
        raise AssertionError("闲聊快路不该触检索")
    monkeypatch.setattr(retriever_mod, "retrieve", boom_retrieve)
    st = _st()
    rec = _Rec()

    async def _run():
        msg = SimpleNamespace(role="user", content="你好")
        return [f async for f in stream_graph_turn(
            [msg], st, _FakeModel(), conversation_id=None, persister=rec)]

    frames = asyncio.run(_run())
    assert frames
    assert rec.final_calls and rec.final_calls[-1]["snap"] is None
