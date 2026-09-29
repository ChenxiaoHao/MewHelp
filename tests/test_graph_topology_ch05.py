"""ch05 Task 3: StateGraph 骨架四出口拓扑 + InMemorySaver 跨轮。

fake model 脚本化 AIMessage(意图 JSON/收敛答案);retrieve 经 monkeypatch
app.workflows.retriever_mod 注入——生产代码用模块属性调用点,测试不打进真 Milvus。
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.rag import retriever as retriever_mod
from app.rag.retriever import RetrieveResult, ScoredRow
from app.workflows.graph import build_graph
from app.workflows.nodes import CHITCHAT_FIXED, COMPLAINT_FIXED


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
        # T5 起 agent 节点走流式:整条脚本消息作单 chunk 透出(聚合语义等价)
        self.calls += 1
        self.last_msgs = list(msgs)
        ai = self.script.pop(0)
        yield AIMessageChunk(
            content=ai.content,
            tool_call_chunks=[
                {"name": t["name"], "args": json.dumps(t["args"], ensure_ascii=False),
                 "id": t["id"], "type": "tool_call", "index": i}
                for i, t in enumerate(ai.tool_calls)
            ],
        )


@pytest.fixture
def fake_settings():
    # T4 起闸节点读 retrieval_low_conf_threshold;T5 起 agent 读 react 双熔断键
    return SimpleNamespace(tool_timeout_seconds=5.0, tool_max_retries=0,
                           retrieval_low_conf_threshold=0.161,
                           max_agent_steps=6, react_token_budget=8000,
                           history_token_budget=4000,
                           # ch08 T7:agent_node 现调 snapshot_tools,死端口桩=恒降级内置面
                           mcp_logistics_url="http://127.0.0.1:9599/mcp",
                           mcp_aftersale_url="http://127.0.0.1:9598/mcp")


@pytest.fixture
def fake_row():
    return SimpleNamespace(category="退款", questions="怎么退款", answer="联系售后退款")


@pytest.fixture
def one_chunk_result(fake_row):
    return RetrieveResult(chunks=[ScoredRow(chunk_id=1, score=0.87, row=fake_row)])


@pytest.fixture
def fake_env(monkeypatch, fake_settings):
    def _install(script, chunks):
        model = ScriptModel(script)
        async def fake_retrieve(query, *, strategy=None, category=None, settings=None, understood=None):
            return RetrieveResult(chunks=list(chunks))
        monkeypatch.setattr(retriever_mod, "retrieve", fake_retrieve)
        graph = build_graph(fake_settings, model)
        return SimpleNamespace(graph=graph, model=model)
    return _install


def init_state(q):
    return {"messages": [HumanMessage(content=q)], "user_query": q}


def thread_cfg(n):
    return {"configurable": {"thread_id": f"t{n}", "conversation_id": n}}


async def test_knowledge_route_forces_retrieval_before_answer(fake_env, one_chunk_result):
    # ch06 T2 重定向:退款退货已改道 refund,知识路探针换「商品咨询」;JSON 补 confidence
    env = fake_env(
        [AIMessage(content='{"intent":"商品咨询","confidence":0.9}'),
         AIMessage(content="政策是7天无理由")],
        one_chunk_result.chunks,
    )
    out = await env.graph.ainvoke(init_state("退款政策是什么"), config=thread_cfg(1))
    assert out["log"]["nodes"] == ["coref", "intent", "retrieve", "gate", "agent", "logging"]  # 顺序示例化
    assert out["log"]["retrieve_hits"] == 1   # 验收1 单测面:检索节点被走到
    assert env.model.calls == 2               # 意图1 + agent收敛1


async def test_chitchat_makes_zero_model_calls(fake_env):
    env = fake_env([], [])
    out = await env.graph.ainvoke(init_state("你好"), config=thread_cfg(2))
    assert env.model.calls == 0 and out["answer_text"] == CHITCHAT_FIXED


async def test_complaint_emits_two_unbound_suggestions(fake_env):
    env = fake_env([AIMessage(content='{"intent":"投诉","confidence":0.95}')], [])
    out = await env.graph.ainvoke(init_state("我要投诉"), config=thread_cfg(3))
    assert [s["action"] for s in out["suggestions"]] == ["transfer_human", "create_ticket"]
    assert env.model.calls == 1  # 仅意图分类;投诉话术固定
    assert out["answer_text"] == COMPLAINT_FIXED


async def test_business_route_skips_retrieval(fake_env):
    # 业务数据类旁路:拓扑证据=日志节点链里没有 retrieve/gate(需求 3/7)
    env = fake_env([AIMessage(content='{"intent":"物流","confidence":0.9}'),
                    AIMessage(content="包裹已到")], [])
    out = await env.graph.ainvoke(init_state("物流到哪了"), config=thread_cfg(4))
    assert out["log"]["nodes"] == ["coref", "intent", "agent", "logging"]
    assert "retrieve" not in out["log"]["nodes"] and "gate" not in out["log"]["nodes"]


async def test_gate_fail_renders_refusal_and_persists(fake_env, fake_settings, monkeypatch):
    from app.services import refusals
    async def no_pool(*a, **k):  # 落池语义归 T4 测管,这里只防真实写库副作用
        pass
    monkeypatch.setattr(refusals, "pool_low_confidence", no_pool)
    env = fake_env([AIMessage(content='{"intent":"商品咨询","confidence":0.9}')], [])  # 空 chunks → 闸不过
    out = await env.graph.ainvoke(init_state("退款政策是什么"), config=thread_cfg(5))
    assert out["log"]["nodes"] == ["coref", "intent", "retrieve", "gate", "logging"]
    assert out["answer_text"] == refusals.REFUSAL_ANSWER


async def test_p2_fallback_to_knowledge_after_retry(fake_env, one_chunk_result):
    env = fake_env([AIMessage(content="not json"), AIMessage(content="still not"),
                    AIMessage(content="兜底回答")], one_chunk_result.chunks)
    out = await env.graph.ainvoke(init_state("随便说点什么"), config=thread_cfg(6))
    assert out["intent"] == "其他"       # ch06 P1:重试×2 仍败 → 归「其他」(route=knowledge)
    assert out["log"]["nodes"] == ["coref", "intent", "retrieve", "gate", "agent", "logging"]
    assert env.model.calls == 3          # 意图×2(含重试) + agent×1


async def test_per_turn_state_does_not_leak_across_turns(fake_env, one_chunk_result):
    """M1-I1:同 thread 三轮(投诉→知识→业务),建议/证据/节点链均按轮隔离;
    messages 历史照常累积(checkpointer 语义不变)。
    ch06 T1 重定向:第 2/3 轮有历史→coref 各多耗一条脚本消息(输出=补全后问法)。"""
    env = fake_env([
        AIMessage(content='{"intent":"投诉","confidence":0.95}'),
        AIMessage(content="退款政策是什么"),
        AIMessage(content='{"intent":"商品咨询","confidence":0.9}'), AIMessage(content="政策A"),
        AIMessage(content="订单1001物流到哪了"),
        AIMessage(content='{"intent":"物流","confidence":0.9}'), AIMessage(content="包裹已到"),
    ], one_chunk_result.chunks)
    out = await env.graph.ainvoke(init_state("我要投诉"), config=thread_cfg(20))
    assert out["suggestions"]  # 第一轮投诉确有建议
    await env.graph.ainvoke(init_state("退款政策是什么"), config=thread_cfg(20))
    out3 = await env.graph.ainvoke(init_state("订单1001物流到哪了"), config=thread_cfg(20))
    assert out3["evidence"] == []        # 业务轮不残留上一知识轮证据
    assert out3["suggestions"] == []     # 投诉轮建议不越轮
    assert out3["log"]["nodes"] == ["coref", "intent", "agent", "logging"]  # 链按轮计
    assert not any(type(m).__name__ == "SystemMessage" and "知识库证据" in m.content
                   for m in env.model.last_msgs)  # 陈旧证据未被注入模型调用


def test_ctx_entry_point_edge_ch07(fake_settings):
    """ch07 T7 随迁(规4):入口从 coref 挪到 ctx——断 __start__→ctx→coref,
    且 coref 不再是入口(旧边必须不存在,否则图面还是双入口)。"""
    from langgraph.graph import START
    g = build_graph(fake_settings, ScriptModel([]))
    edges = {(e.source, e.target) for e in g.get_graph().edges}
    assert (START, "ctx") in edges and ("ctx", "coref") in edges
    assert (START, "coref") not in edges


async def test_coref_completion_keeps_history(fake_env, one_chunk_result):
    """ch06 T1 重定向(原 test_coref_passthrough_keeps_history):首轮零调用,
    次轮 coref 走 LLM(脚本插补全输出),InMemorySaver 跨轮累积语义不变。"""
    env = fake_env([AIMessage(content='{"intent":"商品咨询","confidence":0.9}'),
                    AIMessage(content="政策A"),
                    AIMessage(content="第二问是什么退款政策"),
                    AIMessage(content='{"intent":"商品咨询","confidence":0.9}'),
                    AIMessage(content="政策B")],
                   one_chunk_result.chunks)
    await env.graph.ainvoke(init_state("第一问"), config=thread_cfg(7))
    out = await env.graph.ainvoke(init_state("第二问"), config=thread_cfg(7))
    kinds = [type(m).__name__ for m in out["messages"]]
    assert kinds.count("HumanMessage") >= 2 and kinds.count("AIMessage") >= 2  # InMemorySaver 跨轮累积
