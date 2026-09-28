"""ch07 T7:ctx 入口节点(降级+摘要触发决策)+ coref/agent 换供 store 供史装配。

行为红线(spec「后台异步摘要」+Review Focus 2/3):无 store 轮零副作用;
有 store 轮先降级再判层2 超预算→排任务;store 炸了不拖垮本轮(透传回 {})。
coref 的 {history} 来自 build_history_view(含摘要行;末行当前用户句剔掉——
bootstrap 已先落 user 行,首轮剔完为空=passthrough 语义保留);每轮 history_ctx。
"""

import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.workflows import nodes as N
from app.workflows.graph import build_graph
from app.workflows.nodes import make_agent_node, make_coref_node, make_ctx_node

DEMO = dict(model_context_window=18000, max_output_tokens=2000, max_user_input_tokens=2000,
            max_agent_steps=3, tool_result_max_tokens=1200, rerank_top_k=5,
            turns_to_keep=20, steady_tokens_per_turn=500, assistant_head_chars=60,
            summary_inject_tokens=707, safety_margin_tokens=1000, history_view_messages=6,
            chunk_size=500, retrieval_low_conf_threshold=0.161,
            react_max_iterations=6, react_token_budget=8000,
            tool_timeout_seconds=5.0, tool_max_retries=0, history_token_budget=4000,
            intent_confidence_threshold=0.6, demo_user_id="u_demo")


def _settings(**over):
    return SimpleNamespace(**{**DEMO, **over})


def _row(rid, role, content=None, tool_calls=None, tool_call_id=None):
    return SimpleNamespace(id=rid, role=role, content=content,
                           tool_calls=tool_calls, tool_call_id=tool_call_id)


class FakeStore:
    def __init__(self, rows, ctx, boom=None):
        self.cid = 42
        self.session_factory = lambda: None
        self._rows, self._ctx, self._boom = rows, list(ctx), boom or set()
        self.sets = []

    def _guard(self, name):
        if name in self._boom:
            raise RuntimeError(f"store boom {name}")

    async def load_ctx(self):
        self._guard("load_ctx")
        return tuple(self._ctx)

    async def fetch_all_rows(self):
        self._guard("fetch_all_rows")
        return list(self._rows)

    async def fetch_layer1(self):
        return [r for r in self._rows if r.id > self._ctx[2]]

    async def set_layer1_from(self, value):
        self.sets.append(value)
        self._ctx[2] = value


def _cfg(store):
    return {"configurable": {"thread_id": "conv-42", "conversation_id": 42,
                             "ctx_store": store, "persister": None}}


class ScriptModel:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.last_msgs = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, msgs):
        self.calls += 1
        self.last_msgs = list(msgs)
        return self.script.pop(0)


@pytest.fixture
def _quiet_ctx(monkeypatch):
    """默认把 ctx 的降级/排程换成记录桩,单测各取所需。"""
    rec = []
    async def fake_degrade(store, settings):
        rec.append("degrade")
        return None
    monkeypatch.setattr(N, "degrade_if_needed", fake_degrade)
    monkeypatch.setattr(N, "schedule_summary",
                        lambda *a: rec.append("schedule") or True)
    return rec


async def test_ctx_no_store_zero_side_effects(_quiet_ctx):
    upd = await make_ctx_node(_settings(), ScriptModel([]))(
        {"user_query": "你好"}, {"configurable": {"conversation_id": None}})
    assert upd == {} and _quiet_ctx == []


async def test_ctx_degrade_then_triggers_when_layer2_over_budget(_quiet_ctx):
    big = [_row(i, "user", "汉" * 500) for i in range(1, 9)]  # 层2=(0,8] ≈4032>1695
    store = FakeStore(big, (None, 0, 8))
    upd = await make_ctx_node(_settings(), ScriptModel([]))({}, _cfg(store))
    assert upd == {} and _quiet_ctx == ["degrade", "schedule"]


async def test_ctx_no_trigger_when_layer2_fits(_quiet_ctx):
    store = FakeStore([_row(1, "user", "短")], (None, 0, 1))  # 层2 一条 ≈5 ≤1695
    await make_ctx_node(_settings(), ScriptModel([]))({}, _cfg(store))
    assert _quiet_ctx == ["degrade"]        # 降级照跑,摘要不排(验收3:不压为常态)


async def test_ctx_store_boom_does_not_break_round(_quiet_ctx):
    store = FakeStore([], (None, 0, 0), boom={"load_ctx"})
    upd = await make_ctx_node(_settings(), ScriptModel([]))({}, _cfg(store))
    assert upd == {}                        # 异常被吞进 WARN,本轮照常走图


async def test_coref_history_from_view_with_projection(monkeypatch):
    rows = [_row(1, "user", "先前句"), _row(2, "assistant", "旧答"),
            _row(3, "user", "你好")]        # id3 = bootstrap 落的当前句
    store = FakeStore(rows, ("早前梗概一条", 0, 0))
    model = ScriptModel([AIMessage(content="先前句的后续")])
    # 线程形态=回填后(旧轮+当前句):首轮判定看线程态(Ruling),此测钉 {history} 供体
    upd = await make_coref_node(model, _settings())(
        {"user_query": "你好",
         "messages": [HumanMessage("先前句"), AIMessage("旧答"), HumanMessage("你好")]},
        _cfg(store))
    human = [m for m in model.last_msgs if isinstance(m, HumanMessage)][0].content
    assert "早前梗概一条" in human and "先前句" in human
    assert "用户:你好" not in human          # 当前句剔掉:自己不许当自己的历史
    assert upd["resolved_query"] == "先前句的后续" and upd["log"]["coref"] == "done"


async def test_coref_first_round_with_store_still_passthrough(monkeypatch):
    store = FakeStore([_row(1, "user", "你好")], (None, 0, 0))  # 只有 bootstrap 行
    seen = []
    monkeypatch.setattr(N, "build_history_view",
                        lambda *a: _view("(无)", seen))
    model = ScriptModel([])
    upd = await make_coref_node(model, _settings())(
        {"user_query": "你好", "messages": [HumanMessage("你好")]}, _cfg(store))
    assert upd["log"]["coref"] == "passthrough" and model.calls == 0


def _view(value, sink):
    async def _v(store, settings):
        sink.append("view")
        return value
    return _v()


async def test_history_ctx_logged_on_chitchat_round(monkeypatch, caplog):
    """需求 6:每轮必打——闲聊轮无模型调用也出 history_ctx(store 面)。"""
    store = FakeStore([_row(1, "user", "你好")], (None, 0, 0))
    monkeypatch.setattr(N, "degrade_if_needed", lambda *a: _none())
    monkeypatch.setattr(N, "schedule_summary", lambda *a: True)
    graph = build_graph(_settings(), ScriptModel([]))
    with caplog.at_level(logging.INFO):
        out = await graph.ainvoke(
            {"messages": [HumanMessage("你好")], "user_query": "你好"},
            config=_cfg(store))
    assert out["log"]["nodes"] == ["coref", "intent", "chitchat", "logging"]
    assert any(r.getMessage().startswith("history_ctx cid=42") for r in caplog.records)


async def _none():
    return None


async def test_agent_receives_five_segment_with_injection(monkeypatch, caplog):
    rows = [_row(1, "user", "帮我查库存" + "汉" * 100), _row(2, "user", "那物流呢"),
            _row(3, "assistant", "已发货" + "细" * 300)]
    store = FakeStore(rows, ("第一段梗概", 0, 1))   # 层2=id1,层1=id2..3
    captured = {}

    async def fake_react(state, settings, model, persister=None):
        captured["msgs"] = list(state["messages"])
        captured["state"] = state
        yield ("token", "好")
        yield ("done", {"steps": 1, "suggestions": []})
    monkeypatch.setattr(N, "react_agent_stream", fake_react)
    cur = HumanMessage("我要退款")
    with caplog.at_level(logging.INFO):
        upd = await make_agent_node(ScriptModel([]), _settings())(
            {"messages": [cur], "user_query": "我要退款", "route": "knowledge",
             "evidence": [{"text": "条款X"}], "order_data": {"order_id": "1001"},
             "log": {}}, _cfg(store))
    assert captured["state"]["evidence"] == []      # react 旧 System 前置清空(防双份)
    assert captured["state"]["order_data"] == {}
    assert any(r.getMessage().startswith("model_ctx cid=42")  # 验收4 grep 锚
               for r in caplog.records)
    msgs = captured["msgs"]
    assert isinstance(msgs[0], SystemMessage)
    assert sum(isinstance(m, SystemMessage) for m in msgs) == 1   # 证据不再 System 前置(段5合一)
    inj = msgs[-1]
    assert inj is not cur and isinstance(inj, HumanMessage)
    assert "早前对话梗概" in inj.content and "条款X" in inj.content and "1001" in inj.content
    assert upd["answer_text"] == "好"
