"""ch07 T7:空线程回填(Review Focus 3)——重启/切换后 thread 空但 DB 有史,
coref 不许把续聊误判成首轮;回填形态只有 Human/AI 两类,绝无孤儿 tool 链。"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.workflows import graph as G
from app.workflows import nodes as N
from app.workflows.graph import _refill_input

DEMO = dict(model_context_window=18000, max_output_tokens=2000, max_user_input_tokens=2000,
            max_agent_steps=3, tool_result_max_tokens=1200, rerank_top_k=5,
            turns_to_keep=20, steady_tokens_per_turn=500, assistant_head_chars=60,
            summary_inject_tokens=707, safety_margin_tokens=1000, history_view_messages=2,
            chunk_size=500, retrieval_low_conf_threshold=0.161,
            react_token_budget=8000,
            tool_timeout_seconds=5.0, tool_max_retries=0, history_token_budget=4000,
            intent_confidence_threshold=0.6, demo_user_id="u_demo")


def _settings(**over):
    return SimpleNamespace(**{**DEMO, **over})


def _row(rid, role, content=None, tool_calls=None):
    return SimpleNamespace(id=rid, role=role, content=content, tool_calls=tool_calls)


class FakeStore:
    def __init__(self, rows, ctx):
        self.cid = 7
        self._rows, self._ctx = rows, list(ctx)

    async def load_ctx(self):
        return tuple(self._ctx)

    async def fetch_all_rows(self):
        return list(self._rows)

    async def fetch_layer1(self):
        return [r for r in self._rows if r.id > self._ctx[2]]

    async def set_layer1_from(self, value):
        self._ctx[2] = value


class FakeGraph:
    def __init__(self, messages):
        self._values = {"messages": messages} if messages is not None else {}

    async def aget_state(self, cfg):
        return SimpleNamespace(values=self._values)


def _cfg():
    return {"configurable": {"thread_id": "conv-7"}}


async def test_refill_only_human_ai_no_tool_rows():
    rows = [_row(1, "user", "旧问"),
            _row(2, "assistant", None, [{"id": "c1", "name": "q", "args": {}}]),
            _row(3, "tool", "结果"),
            _row(4, "assistant", "旧答"),
            _row(5, "user", "你好")]      # 末行=bootstrap 刚落的当前句
    store = FakeStore(rows, (None, 0, 0))
    prefix = await _refill_input(FakeGraph([]), _cfg(), store, _settings(), "你好")
    assert [type(m) for m in prefix] == [HumanMessage, AIMessage]   # tool_calls 行/tool 行/当前句全剔
    assert [m.content for m in prefix] == ["旧问", "旧答"]
    assert not any(isinstance(m, ToolMessage) for m in prefix)
    assert not any(getattr(m, "tool_calls", None) for m in prefix)


async def test_refill_caps_at_history_view_messages_times_two():
    rows = [_row(i, "user" if i % 2 else "assistant", f"第{i}句") for i in range(1, 21)]
    rows.append(_row(21, "user", "你好"))
    store = FakeStore(rows, (None, 0, 0))
    prefix = await _refill_input(FakeGraph([]), _cfg(), store, _settings(), "你好")
    assert len(prefix) == 4               # history_view_messages=2 → 近 4 条
    assert prefix[-1].content == "第20句"  # 当前句不混进前缀


async def test_refill_skipped_when_thread_nonempty():
    store = FakeStore([_row(1, "user", "旧问")], (None, 0, 0))
    alive = [HumanMessage("在途历史")]
    assert await _refill_input(FakeGraph(alive), _cfg(), store, _settings(), "你好") == []


async def test_stream_turn_refills_empty_thread_and_coref_not_first_round(monkeypatch):
    """空 thread + DB 有史 → 回填进图 → coref 见史走 LLM(不误判首轮)。"""
    G.reset_checkpointer()
    rows = [_row(1, "user", "我买的台灯坏了"), _row(2, "assistant", "请问维修还是换货"),
            _row(3, "user", "你好")]
    store = FakeStore(rows, ("旧梗概甲", 0, 0))   # 梗概行=DB view 专属证据(线程回退给不出)
    model = SimpleNamespace(
        calls=[],
        bind_tools=lambda tools: None,
    )
    async def spy_ainvoke(msgs):
        model.calls.append(msgs)
        return AIMessage(content="台灯坏了想换货")
    model.ainvoke = spy_ainvoke
    monkeypatch.setattr(N, "degrade_if_needed", lambda *a: _none())
    monkeypatch.setattr(N, "schedule_summary", lambda *a: True)
    frames = [f async for f in G.stream_graph_turn(
        [SimpleNamespace(content="你好")], _settings(), model,
        conversation_id=7, persister=None, ctx_store=store)]
    assert len(model.calls) == 1, "回填后 coref 走 LLM 恰一次(闲聊快路抢在判类前,不再调模型)"
    human = [m for m in model.calls[0] if isinstance(m, HumanMessage)][0].content
    assert "旧梗概甲" in human              # 史来自 DB view:梗概行是线程回退给不出的专属证据
    assert "请问维修还是换货" in human       # 窗=2 → 层1 末 2 行(剔当前句后仍有客服行)
    assert N.chitchat_fast_path("你好")   # 意图快路:判类零模型调用,coref 是唯一一次
    assert len(model.calls) == 1
    assert ("token", N.CHITCHAT_FIXED) in frames   # 闲聊固定话术照常整段补发


async def _none():
    return None
