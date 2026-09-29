"""ch06 Task 4: 退款确定性子流程五节点 + 续跑识别 + gate source + refund_apply + 订单注入。

图级用例复用 ScriptModel(ch05 fake_env 同款)+ monkeypatch retriever_mod.retrieve;
react 注入用例直调 react_agent_stream——ch07 T8 起注入上移段5 装配,react 层反向守
「任何 System 前置都不再出现」(正向面归 test_ctx_node_ch07 五段装配测)。
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.react import react_agent_stream
from app.rag import retriever as retriever_mod
from app.rag.retriever import RetrieveResult, ScoredRow
from app.services import refusals
from app.workflows import nodes as N
from app.workflows.graph import build_graph
from app.workflows.nodes import CHITCHAT_FIXED
from app.workflows.state import REFUND_APPLY, SELECT_ORDER_ASK, TRANSFER_HUMAN


class ScriptModel:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, msgs):
        self.calls += 1
        self.last_msgs = list(msgs)
        return self.script.pop(0)

    async def astream(self, msgs):
        self.calls += 1
        self.last_msgs = list(msgs)
        ai = self.script.pop(0)
        yield AIMessageChunk(content=ai.content)


@pytest.fixture
def fake_settings():
    return SimpleNamespace(tool_timeout_seconds=5.0, tool_max_retries=0,
                           retrieval_low_conf_threshold=0.161,
                           max_agent_steps=6, react_token_budget=8000,
                           history_token_budget=4000, demo_user_id="demo_user",
                           rerank_top_n=10, intent_small_model="",
                           intent_confidence_threshold=0.75,
                           # ch08 T7:agent_node 现调 snapshot_tools,死端口桩=恒降级内置面
                           mcp_logistics_url="http://127.0.0.1:9599/mcp",
                           mcp_aftersale_url="http://127.0.0.1:9598/mcp")


@pytest.fixture
def fake_row():
    return SimpleNamespace(category="退款", questions="怎么退款", answer="联系售后退款")


def _chunks(fake_row):
    return [ScoredRow(chunk_id=1, score=0.9, row=fake_row),
            ScoredRow(chunk_id=2, score=0.7, row=fake_row)]


@pytest.fixture
def fake_env(monkeypatch, fake_settings, fake_row):
    def _install(script, chunks=None):
        model = ScriptModel(script)
        calls = {"retrieve": 0}

        async def fake_retrieve(query, *, strategy=None, category=None,
                                settings=None, understood=None):
            calls["retrieve"] += 1
            return RetrieveResult(chunks=list(chunks if chunks is not None
                                              else _chunks(fake_row)))
        monkeypatch.setattr(retriever_mod, "retrieve", fake_retrieve)
        async def no_pool(*a, **k):
            pass
        monkeypatch.setattr(refusals, "pool_low_confidence", no_pool)
        # 每测独立 checkpointer:生产共享实例会让本文件 t1–t7 线程号
        # 撞上 test_graph_topology_ch05 的同名线程(全量序下历史串轮)
        graph = build_graph(fake_settings, model, checkpointer=InMemorySaver())
        return SimpleNamespace(graph=graph, model=model, retrieve_calls=calls)
    return _install


def init_state(q):
    return {"messages": [HumanMessage(content=q)], "user_query": q}


def thread_cfg(n):
    return {"configurable": {"thread_id": f"t{n}", "conversation_id": n}}


REFUND_JSON = '{"intent":"退款退货","confidence":0.95}'


async def test_no_order_pops_selector_with_zero_retrieval(fake_env):
    env = fake_env([AIMessage(content=REFUND_JSON)])
    out = await env.graph.ainvoke(init_state("我要退款"), config=thread_cfg(1))
    assert out["log"]["nodes"] == ["coref", "intent", "refund_slot",
                                   "refund_selector", "logging"]
    assert [c["order_id"] for c in out["orders_payload"]] == ["1001", "1002", "1003"]
    assert out["pending_flow"] == "refund"
    assert out["answer_text"] == SELECT_ORDER_ASK
    assert env.retrieve_calls["retrieve"] == 0      # 选择器轮零检索零 Agent
    assert env.model.calls == 1                     # 仅意图一次


async def test_selection_resumes_subflow_single_path(fake_env):
    env = fake_env([
        AIMessage(content=REFUND_JSON),                       # 轮1 意图
        AIMessage(content='{"queries": ["多久内可退", "退款运费谁承担"]}'),  # 轮2 扩写
        AIMessage(content="这一单可以申请退款"),               # 轮2 Agent
    ])
    await env.graph.ainvoke(init_state("我要退款"), config=thread_cfg(2))
    before = env.model.calls
    out = await env.graph.ainvoke(init_state("我选择订单 1001"), config=thread_cfg(2))
    assert env.model.calls - before == 2   # 续跑轮只有 扩写+Agent:coref/intent 零调用
    nodes = out["log"]["nodes"]
    for n in ("refund_fetch", "refund_expand", "refund_policy", "refund_gate", "agent"):
        assert n in nodes
    assert out["slot_order_id"] == "1001"
    assert out["order_data"]["order_id"] == "1001"
    assert out["pending_flow"] == ""       # 读后即清
    assert out["expanded_queries"] == ["订单 1001 能不能申请退款？",
                                       "多久内可退", "退款运费谁承担"]  # 原问法居首
    assert out["resolved_query"] == "订单 1001 能不能申请退款？"
    assert [(c["chunk_id"], c["score"]) for c in out["evidence"]] == [(1, 0.9), (2, 0.7)]
    assert env.retrieve_calls["retrieve"] == 3   # 每查询一路,选择器轮零检索
    assert out["log"].get("resume") is True


async def test_expand_garbage_degrades_to_single_query(fake_env):
    env = fake_env([AIMessage(content=REFUND_JSON),
                    AIMessage(content="不配合的扩写输出"),
                    AIMessage(content="可以退")])
    out = await env.graph.ainvoke(init_state("订单1001我要退"), config=thread_cfg(3))
    assert out["expanded_queries"] == ["订单1001我要退"]   # 降级单路不阻断
    assert out["log"]["nodes"].count("agent") == 1


async def test_policy_empty_refuses_without_refund_apply(fake_env):
    env = fake_env([AIMessage(content=REFUND_JSON),
                    AIMessage(content='{"queries": ["退款时限"]}')], chunks=[])
    out = await env.graph.ainvoke(init_state("订单1001我要退"), config=thread_cfg(4))
    assert out["log"]["nodes"] == ["coref", "intent", "refund_slot", "refund_fetch",
                                   "refund_expand", "refund_policy", "refund_gate",
                                   "logging"]
    assert out["answer_text"] == refusals.REFUSAL_ANSWER
    assert [s["action"] for s in out["suggestions"]] == ["transfer_human"]  # 无 refund_apply


async def test_gate_pass_attaches_refund_apply(fake_env):
    env = fake_env([AIMessage(content=REFUND_JSON),
                    AIMessage(content='{"queries": ["退款时限"]}'),
                    AIMessage(content="符合政策,可退")])
    out = await env.graph.ainvoke(init_state("订单1001我要退"), config=thread_cfg(5))
    assert out["suggestions"] == [REFUND_APPLY]
    assert out["log"]["gate_pass"] is True


async def test_budget_transfer_suggestion_not_overridden(fake_env, monkeypatch):
    async def fake_react(state, settings, model, *, persister=None, specs=None):
        # ch08 T7:agent_node 新增 specs kwarg(快照全集),替身同签名
        yield ("token", "预算用尽收尾")
        yield ("done", {"steps": 6, "suggestions": [TRANSFER_HUMAN]})
    monkeypatch.setattr(N, "react_agent_stream", fake_react)
    env = fake_env([AIMessage(content=REFUND_JSON),
                    AIMessage(content='{"queries": ["退款时限"]}')])
    out = await env.graph.ainvoke(init_state("订单1001我要退"), config=thread_cfg(6))
    assert [s["action"] for s in out["suggestions"]] == ["transfer_human"]


async def test_pending_not_hijack_normal_turn(fake_env):
    env = fake_env([
        AIMessage(content=REFUND_JSON),          # 轮1 意图
        AIMessage(content="你好"),               # 轮2 coref(有历史)
    ])
    await env.graph.ainvoke(init_state("我要退款"), config=thread_cfg(7))
    out = await env.graph.ainvoke(init_state("你好"), config=thread_cfg(7))
    assert out["answer_text"] == CHITCHAT_FIXED  # 快路闲聊(Review Focus 2)
    assert out["log"].get("fast_path") is True
    assert out["pending_flow"] == ""             # 读后即清,不劫持
    assert "refund_selector" not in out["log"]["nodes"]


async def test_react_no_longer_prepends_system_injection(fake_settings, fake_row):
    """ch07 T8 断言翻转(规4 随本任务):order_data/evidence 的注入上移到段5 装配
    (build_model_context,当前句后一条 Human),react 层不再产任何 System 前置——
    正向注入面由 tests/test_ctx_node_ch07.py::test_agent_receives_five_segment 钉。
    本测守反向红线:react 收到带 order_data/evidence 的 state 也必须零 System。"""
    class OneShot:
        def bind_tools(self, tools):
            self.last_msgs = None
            return self

        async def astream(self, msgs):
            self.last_msgs = list(msgs)
            yield AIMessageChunk(content="好的")

    m = OneShot()
    st = {"messages": [HumanMessage(content="这单能退吗")],
          "order_data": {"order_id": "1001", "status": "运输中", "amount": 99.0}}
    async for _ in react_agent_stream(st, fake_settings, m):
        pass
    assert not any(isinstance(x, SystemMessage) for x in m.last_msgs)
    assert isinstance(m.last_msgs[0], HumanMessage)      # 入参原样透传

    m2 = OneShot()
    st2 = {"messages": [HumanMessage(content="退货政策是什么")],
           "evidence": [{"chunk_id": 1, "score": 0.9, "text": "7 天"}]}
    async for _ in react_agent_stream(st2, fake_settings, m2):
        pass
    assert not any(isinstance(x, SystemMessage) for x in m2.last_msgs)
