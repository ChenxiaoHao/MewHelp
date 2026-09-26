"""StateGraph 装配（ch05 spec「总体架构与图拓扑」节）。

指代消解→意图识别→按意图分流(knowledge/retrieve→gate/agent|data→agent|
complaint|chitchat)→日志→END。checkpointer=InMemorySaver,仅进程内跨轮(D3)。
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from app.workflows import nodes as N
from app.workflows.state import ChatState


def build_graph(settings, model):
    g = StateGraph(ChatState)
    g.add_node("coref", N.coref_node)
    g.add_node("intent", N.make_intent_node(model))
    g.add_node("retrieve", N.make_knowledge_retrieve_node(settings))
    g.add_node("gate", N.make_confidence_gate_node(settings))
    g.add_node("agent", N.make_agent_node(model, settings))
    g.add_node("complaint", N.complaint_node)
    g.add_node("chitchat", N.chitchat_node)
    g.add_node("logging", N.logging_node)

    g.set_entry_point("coref")
    g.add_edge("coref", "intent")
    # 分流:意图 → 四出口(需求 3)
    g.add_conditional_edges("intent", _dispatch, {
        "knowledge": "retrieve", "data": "agent",
        "complaint": "complaint", "chitchat": "chitchat",
    })
    # 知识闸:证据弱直接兜底不进 Agent(需求 7)
    g.add_edge("retrieve", "gate")
    g.add_conditional_edges("gate", _after_gate, {"pass": "agent", "fail": "logging"})
    g.add_edge("agent", "logging")
    g.add_edge("complaint", "logging")
    g.add_edge("chitchat", "logging")
    g.add_edge("logging", END)
    return g.compile(checkpointer=InMemorySaver())


def _dispatch(state):
    return state.get("route", "knowledge")


def _after_gate(state):
    return "pass" if state.get("gate_pass") else "fail"


_graph = None


def get_graph(settings=None, model=None):
    """模块级单例入口:Task 6 接线时以真 settings/model 首次构建。"""
    global _graph
    if _graph is None:
        assert settings is not None and model is not None, "首次构建需 settings+model"
        _graph = build_graph(settings, model)
    return _graph
