"""ch05 Task 4: 置信度闸阈值版(P1 纯阈值)+ 弱证据落低置信池。

阈值复用 ch04 闸1 终值 retrieval_low_conf_threshold(0.161,不新造配置键);
落池唯一写方=本闸(知识类预检索不经 query_faq 工具,与闸1 无重叠)。
Review Focus 5:池写失败只 WARN,兜底话术照常。
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.rag import retriever as retriever_mod
from app.rag.retriever import RetrieveResult, ScoredRow
from app.services import refusals
from app.workflows.graph import build_graph
from app.workflows.nodes import evidence_gate_verdict


class ScriptModel:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, msgs):
        self.calls += 1
        return self.script.pop(0)


def _chunks(scores):
    row = SimpleNamespace(category="退款", questions="怎么退款", answer="政策")
    return [ScoredRow(chunk_id=i + 1, score=s, row=row) for i, s in enumerate(scores)]


# ---- 纯函数三组 ----

def test_verdict_high_score_passes():
    ok, best = evidence_gate_verdict(_chunks([0.05, 0.87]), 0.161)
    assert ok is True and best == pytest.approx(0.87)


def test_verdict_low_score_fails():
    ok, _ = evidence_gate_verdict(_chunks([0.05, 0.1]), 0.161)
    assert ok is False


def test_verdict_empty_evidence_fails():
    ok, _ = evidence_gate_verdict([], 0.161)
    assert ok is False


# ---- 节点级(graph 内跑) ----

@pytest.fixture
def gate_env(monkeypatch):
    def _install(scores, script, real_pool=False):
        st = SimpleNamespace(
            tool_timeout_seconds=5.0, tool_max_retries=0,
            retrieval_low_conf_threshold=0.161,
        )
        async def fake_retrieve(query, *, strategy=None, category=None, settings=None, understood=None):
            return RetrieveResult(chunks=_chunks(scores))
        monkeypatch.setattr(retriever_mod, "retrieve", fake_retrieve)
        calls = []
        if not real_pool:
            async def fake_pool(conversation_id, raw_question, source, reason):
                calls.append({"cid": conversation_id, "q": raw_question, "src": source})
            monkeypatch.setattr(refusals, "pool_low_confidence", fake_pool)
        model = ScriptModel(script)
        graph = build_graph(st, model)
        return graph, model, calls
    return _install


def cfg(n=11):
    return {"configurable": {"thread_id": f"g{n}", "conversation_id": n}}


def init_state(q):
    return {"messages": [HumanMessage(content=q)], "user_query": q}


async def test_weak_evidence_refuses_without_agent_and_pools(gate_env):
    graph, model, pools = gate_env([0.05], [AIMessage(content='{"intent":"退款退货"}')])
    out = await graph.ainvoke(init_state("退款政策"), config=cfg())
    assert out["log"]["nodes"] == ["coref", "intent", "retrieve", "gate", "logging"]
    assert out["answer_text"] == refusals.REFUSAL_ANSWER
    assert model.calls == 1  # 未进 Agent
    assert pools == [{"cid": 11, "q": "退款政策", "src": "ch05_gate"}]
    # brief Produces:弱证据附转人工建议(两按钮独立性归 T8,前端自选)
    assert out["suggestions"] == [{"action": "transfer_human", "label": "转人工"}]


async def test_strong_evidence_enters_agent(gate_env):
    graph, model, pools = gate_env(
        [0.9], [AIMessage(content='{"intent":"退款退货"}'), AIMessage(content="7天无理由")])
    out = await graph.ainvoke(init_state("退款政策"), config=cfg(12))
    assert out["gate_pass"] is True and out["answer_text"] == "7天无理由"
    assert pools == []  # 证据够不落池


async def test_rrf_degraded_scores_bypass_threshold(gate_env):
    """M1-I3:ch04 重排降级路径返回 RRF 分(rrf_k=60 双路上限≈0.033,与 0.161
    阈值不可通约)→ 该带内非空证据旁路过闸进 Agent(对齐闸1「降级跳过」决定)。"""
    graph, model, pools = gate_env(
        [0.03], [AIMessage(content='{"intent":"退款退货"}'), AIMessage(content="降级也可答")])
    out = await graph.ainvoke(init_state("退款政策"), config=cfg(14))
    assert out["gate_pass"] is True and out["log"]["gate_scale"] == "rrf_degraded"
    assert out["answer_text"] == "降级也可答"
    assert pools == []  # 旁路不落池


async def test_pool_write_failure_does_not_block_refusal(gate_env, monkeypatch):
    """Review Focus 5 真路径:不 patch 包装函数,让真 pool_low_confidence
    在建 session 时炸(engine 未初始化=无 MySQL 环境),其内部 try/except 吞掉,
    兜底话术照常到达。"""
    def boom():
        raise RuntimeError("engine not configured")
    monkeypatch.setattr(refusals, "get_session_factory", boom)
    graph, _, _ = gate_env([0.05], [AIMessage(content='{"intent":"退款退货"}')], real_pool=True)
    out = await graph.ainvoke(init_state("退款政策"), config=cfg(13))
    assert out["answer_text"] == refusals.REFUSAL_ANSWER
