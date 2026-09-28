"""StateGraph 装配（ch05 spec「总体架构与图拓扑」节）。

指代消解→意图识别→按意图分流(knowledge/retrieve→gate/agent|data→agent|
refund/槽位检→选择器|取单→扩写→政策→gate→agent|complaint|chitchat)→日志→END。
checkpointer=InMemorySaver,仅进程内跨轮(D3)。

T6 起本模块另负责 `stream_graph_turn` 适配层：图流(custom 模式透传的 ReAct
事件 + 终态) → routes 直接消费的 ToolEvent 形状帧流。
"""

import logging
import uuid
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from app.services.chat_service import to_langchain_messages
from app.workflows import nodes as N
from app.workflows.state import ChatState

logger = logging.getLogger(__name__)

# 进程内共享 checkpointer：图按请求现装（模型/settings 可注入测试），
# 线程态跨轮保留靠这唯一实例；重启即丢是 D3 明示语义。
_checkpointer = InMemorySaver()


def reset_checkpointer() -> None:
    """清空全部线程态（测试隔离用；生产无调用点）。"""
    global _checkpointer
    _checkpointer = InMemorySaver()


def build_graph(settings, model, checkpointer=None):
    g = StateGraph(ChatState)
    g.add_node("ctx", N.make_ctx_node(settings, model))          # ch07 入口:边界决策
    g.add_node("coref", N.make_coref_node(model, settings))
    g.add_node("intent", N.make_intent_node(model, settings))
    g.add_node("retrieve", N.make_knowledge_retrieve_node(settings))
    g.add_node("gate", N.make_confidence_gate_node(settings))
    g.add_node("agent", N.make_agent_node(model, settings))
    g.add_node("complaint", N.complaint_node)
    g.add_node("chitchat", N.chitchat_node)
    g.add_node("logging", N.logging_node)
    # ch06 退款确定性子流程(需求 5/6):取单→扩写→政策多路检索→闸(复用 ch05 语义)
    g.add_node("refund_slot", N.refund_slot_node)
    g.add_node("refund_selector", N.make_refund_selector_node(settings))
    g.add_node("refund_fetch", N.refund_fetch_node)
    g.add_node("refund_expand", N.make_refund_expand_node(model))
    g.add_node("refund_policy", N.make_refund_policy_node(settings))
    g.add_node("refund_gate", N.make_confidence_gate_node(
        settings, source="ch06_refund_gate", name="refund_gate"))

    g.set_entry_point("ctx")
    g.add_edge("ctx", "coref")
    g.add_edge("coref", "intent")
    # 分流:意图 → 五出口(ch06 需求 3/5;过渡守卫已拆,ledger 记账)
    g.add_conditional_edges("intent", _dispatch, {
        "knowledge": "retrieve", "data": "agent", "refund": "refund_slot",
        "complaint": "complaint", "chitchat": "chitchat",
    })
    # 知识闸:证据弱直接兜底不进 Agent(需求 7)
    g.add_edge("retrieve", "gate")
    g.add_conditional_edges("gate", _after_gate, {"pass": "agent", "fail": "logging"})
    # 退款链:槽位缺 → 弹卡(直接关流);有单号 → 取单→扩写→政策→闸
    g.add_conditional_edges("refund_slot", _after_slot, {
        "fetch": "refund_fetch", "selector": "refund_selector",
    })
    g.add_edge("refund_selector", "logging")
    g.add_edge("refund_fetch", "refund_expand")
    g.add_edge("refund_expand", "refund_policy")
    g.add_edge("refund_policy", "refund_gate")
    g.add_conditional_edges("refund_gate", _after_gate,
                            {"pass": "agent", "fail": "logging"})
    g.add_edge("agent", "logging")
    g.add_edge("complaint", "logging")
    g.add_edge("chitchat", "logging")
    g.add_edge("logging", END)
    return g.compile(checkpointer=checkpointer or _checkpointer)


def _dispatch(state):
    return state.get("route", "knowledge")


def _after_slot(state):
    return "fetch" if state.get("slot_order_id") else "selector"


def _after_gate(state):
    return "pass" if state.get("gate_pass") else "fail"


async def _refill_input(graph, cfg, store, settings, current: str) -> list:
    """空线程回填(Review Focus 3):重启/切换后 thread 无 messages 而 DB 有史 →
    层1 原文近 history_view_messages*2 条前置入图 input。只取 Human/AI 两类:
    tool 行与带 tool_calls 的 assistant 行绝不入(回填形态无工具链,孤儿链=400)。
    """
    state = await graph.aget_state(cfg)
    if ((getattr(state, "values", None) or {}).get("messages") or []):
        return []                                   # 线程非空:checkpoint 即史
    rows = await store.fetch_layer1()
    kept = [r for r in rows
            if r.role == "user" or (r.role == "assistant" and not r.tool_calls)]
    if kept and kept[-1].role == "user" and (kept[-1].content or "") == current:
        kept = kept[:-1]                            # 当前句由调用方尾置,不重复
    return [HumanMessage(r.content or "") if r.role == "user" else AIMessage(r.content or "")
            for r in kept[-(settings.history_view_messages * 2):]]


async def stream_graph_turn(
    chat_messages, settings, model, *,
    conversation_id=None, persister=None, ctx_store=None,
) -> AsyncIterator[tuple]:
    """一轮图编排 → ("token"|"tool_call"|"tool_result"|"suggestions", payload) 帧流。

    thread key=conv-{cid};降级(cid=None,引擎未初始化)→ anon-uuid 线程跨轮不可
    复用,故整包客户端历史入图(与 ch01 无状态语义对齐)。有 cid 时只入本轮新
    human 消息,历史由 checkpointer 按线程累积(spec「State 贯穿」);ch07:线程空
    而 DB 有史(重启/切换)→ ctx_store 回填近史前缀;store 经 configurable 供
    ctx/coref/agent 三消费面。
    落库挂点:Agent 出口由 react 内部三挂点自落(ch04 同源);其余出口在此
    补 on_final_answer——user 行始终归 routes bootstrap。
    """
    graph = build_graph(settings, model)
    current = chat_messages[-1].content
    cfg = {"configurable": {"thread_id": None, "conversation_id": conversation_id,
                            "persister": persister, "ctx_store": ctx_store}}
    if conversation_id is not None:
        cfg["configurable"]["thread_id"] = thread = f"conv-{conversation_id}"
        input_msgs = [HumanMessage(content=current)]
        if ctx_store is not None:
            # 终审 I2:回填=第四处 store 消费面,漏在 T7「store 面异常全吞」兜底
            # 清单外。DB 闪断等价 ch06 无 store 行为(无回填裸进),聊天不断线。
            try:
                prefix = await _refill_input(graph, cfg, ctx_store, settings, current)
            except Exception:  # noqa: BLE001
                logger.warning("refill degraded cid=%s", conversation_id, exc_info=True)
                prefix = []
            input_msgs = prefix + input_msgs
    else:
        cfg["configurable"]["thread_id"] = thread = f"anon-{uuid.uuid4()}"
        input_msgs = to_langchain_messages(chat_messages)
    streamed = False
    final: dict = {}
    async for mode, chunk in graph.astream(
            {"messages": input_msgs,
             "user_query": chat_messages[-1].content},
            config=cfg, stream_mode=["custom", "values"]):
        if mode == "custom":
            kind = chunk[0]
            if kind == "token":
                streamed = True
            if kind in ("token", "tool_call", "tool_result"):
                yield chunk          # done 等内部事件不外发
        elif mode == "values":
            final = chunk
    answer = final.get("answer_text") or ""
    if answer and not streamed:      # 固定话术出口:无流式 token 时整段补发
        yield ("token", answer)
    if answer and "agent" not in (final.get("log") or {}).get("nodes", []):
        try:
            if persister is not None:
                await persister.on_final_answer(conversation_id, answer)
        except Exception:  # noqa: BLE001 —— 落库失败只降级,不挡关流(spec §9)
            logger.warning("persist final answer failed (graph turn)", exc_info=True)
    if final.get("orders_payload"):  # ch06 P7:选择器卡片帧,仅非空才发(防御空帧)
        yield ("orders", {"items": final["orders_payload"]})
    if final.get("suggestions"):     # RF4:末 token 之后、done 之前
        yield ("suggestions", {"items": final["suggestions"]})
