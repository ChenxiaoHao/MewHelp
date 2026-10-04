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
            react_token_budget=8000,
            tool_timeout_seconds=5.0, tool_max_retries=0, history_token_budget=4000,
            intent_confidence_threshold=0.6, demo_user_id="u_demo",
            # ch08 T7:agent_node 现调 snapshot_tools,死端口桩=恒降级内置面
            mcp_logistics_url="http://127.0.0.1:9599/mcp",
            mcp_aftersale_url="http://127.0.0.1:9598/mcp")


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


async def test_ctx_degrade_then_triggers_when_layer2_over_budget(_quiet_ctx, caplog):
    big = [_row(i, "user", "汉" * 500) for i in range(1, 9)]  # 层2=(0,8] ≈4032>1695
    store = FakeStore(big, (None, 0, 8))
    with caplog.at_level(logging.INFO):
        upd = await make_ctx_node(_settings(), ScriptModel([]))({}, _cfg(store))
    assert upd == {} and _quiet_ctx == ["degrade", "schedule"]
    # spec grep 锚「summary trigger 层2 约N token > 预算M」(T11 C1 取证行,补钉)
    assert any(r.getMessage().startswith("summary trigger cid=42")
               for r in caplog.records)


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


async def test_legacy_assembly_still_grounds_evidence(monkeypatch):
    """M3-I2 修复钉:legacy 面(无 store/装配退位)grounding 不丢。

    spec 范围红线 L11「cid=None 降级路径行为与现状一致」——ch06 现状=证据文本
    必入模型输入(System 旧形已按段5 作废,形可变、有无不可变)。"""
    captured = {}

    async def fake_react(state, settings, model, persister=None, specs=None):
        # ch08 T7:agent_node 新增 specs kwarg(快照全集),替身同签名
        captured["msgs"] = list(state["messages"])
        yield ("token", "好")
        yield ("done", {"steps": 1, "suggestions": []})
    monkeypatch.setattr(N, "react_agent_stream", fake_react)
    upd = await make_agent_node(ScriptModel([]), _settings())(
        {"messages": [HumanMessage("退货运费谁承担")], "user_query": "退货运费谁承担",
         "route": "knowledge", "evidence": [{"text": "七天无理由退,运费由商家承担"}],
         "order_data": {"order_id": "1001"}, "log": {}},
        {"configurable": {"conversation_id": None, "ctx_store": None, "persister": None}})
    msgs = captured["msgs"]
    inj = msgs[-1]
    assert isinstance(inj, HumanMessage) and "七天无理由" in inj.content and "1001" in inj.content
    assert sum(isinstance(m, SystemMessage) for m in msgs) == 1   # 人设唯一,证据不 System 前置
    assert upd["answer_text"] == "好"


async def test_agent_receives_five_segment_with_injection(monkeypatch, caplog):
    rows = [_row(1, "user", "帮我查库存" + "汉" * 100), _row(2, "user", "那物流呢"),
            _row(3, "assistant", "已发货" + "细" * 300)]
    store = FakeStore(rows, ("第一段梗概", 0, 1))   # 层2=id1,层1=id2..3
    captured = {}

    async def fake_react(state, settings, model, persister=None, specs=None):
        # ch08 T7:agent_node 新增 specs kwarg(快照全集),替身同签名
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


async def test_five_seg_snapshot_survives_fuse(monkeypatch):
    """M2-C-1:五段装配生产形状(store 在位)快照不随保险丝蒸发。

    链:生产路由有 session 有 cid → ctx_store 非 None → five_seg=True →
    react_input.evidence 清空。若 react 从清空后的 evidence 现算快照,知识轮
    终答行 retrieval_snapshot 恒 NULL——T6 👎 回捞数据源在生产形状整条失效。
    钉法:agent_node 进保险丝前算好 retrieval_snapshot 挂 react_input 显式键。"""
    rows = [_row(1, "user", "偏远地区有附加费吗" + "费" * 100),
            _row(2, "assistant", "有的" + "费" * 300)]
    store = FakeStore(rows, ("第一段梗概", 0, 1))
    captured = {}

    async def fake_react(state, settings, model, persister=None, specs=None):
        captured["state"] = dict(state)
        yield ("token", "好")
        yield ("done", {"steps": 1, "suggestions": []})
    monkeypatch.setattr(N, "react_agent_stream", fake_react)
    await make_agent_node(ScriptModel([]), _settings())(
        {"messages": [HumanMessage("偏远地区有附加费吗")],
         "user_query": "偏远地区有附加费吗", "route": "knowledge",
         "evidence": [{"chunk_id": 41, "score": 0.93, "text": "附加费说明"}],
         "order_data": {"order_id": "1001"}, "log": {}}, _cfg(store))
    assert captured["state"]["evidence"] == []          # 保险丝语义不动
    snap = captured["state"].get("retrieval_snapshot")
    assert snap == [{"chunk_id": 41, "score": 0.93, "text": "附加费说明"}], \
        "五段轮快照须经显式键随 react_input 交下去(👎 回捞生产形状)"
