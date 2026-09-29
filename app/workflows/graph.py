"""StateGraph 装配（ch05 spec「总体架构与图拓扑」节）。

指代消解→意图识别→按意图分流(knowledge/retrieve→gate/agent|data→agent|
refund/槽位检→选择器|取单→扩写→政策→gate→agent|complaint|chitchat)→日志→END。
checkpointer=InMemorySaver,仅进程内跨轮(D3)。

T6 起本模块另负责 `stream_graph_turn` 适配层：图流(custom 模式透传的 ReAct
事件 + 终态) → routes 直接消费的 ToolEvent 形状帧流。
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

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
    _thread_locks.clear()


# 终审 I-1:同线程并发 graph run 竞态——confirm resume(A)与新消息轮隐式
# cancel drain(B)各自「查 pending→重放 interrupt 点」,A/B 交错=同一点双建单
# +「成功」「权限拒绝」终局并存(M2-I3 只封了 confirm↔confirm 方向)。
# 修法:per-thread 锁把「查 pending+drain」「查 pending+resume 头段」串进
# 同一临界区;锁在首批帧 yield 前释放(pending 判定完成后无并发危险面),
# 慢客户端消费不持锁。
_thread_locks: dict[str, asyncio.Lock] = {}


def thread_lock(thread_key: str) -> asyncio.Lock:
    """测试隔离经 reset_checkpointer 清空;残留未用锁在事件循环关闭后
    acquire 会 RuntimeError(resume 调用方抛错=卡片可重试语义,无害)。"""
    lock = _thread_locks.get(thread_key)
    if lock is None:
        lock = _thread_locks.setdefault(thread_key, asyncio.Lock())
    return lock


async def _empty_aiter() -> AsyncIterator:
    """resume 复查扑空面的零帧流。"""
    return
    yield  # pragma: no cover


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
    g.add_node("ticket_confirm", N.ticket_confirm_node)   # ch08 T7 确认流
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
    # ch08 T7:agent 出口二分支——建单提案捕获过就走确认节点(interrupt 暂停),
    # 其余照旧直达 logging。
    g.add_conditional_edges("agent", _after_agent,
                            {"confirm": "ticket_confirm", "done": "logging"})
    g.add_edge("ticket_confirm", "logging")
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


def _after_agent(state):
    return "confirm" if state.get("ticket_preview") else "done"


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
    conversation_id=None, persister=None, ctx_store=None, resume_value=None,
) -> AsyncIterator[tuple]:
    """一轮图编排 → ("token"|"tool_call"|"tool_result"|"ticket_preview"|"suggestions", payload) 帧流。

    thread key=conv-{cid};降级(cid=None,引擎未初始化)→ anon-uuid 线程跨轮不可
    复用,故整包客户端历史入图(与 ch01 无状态语义对齐)。有 cid 时只入本轮新
    human 消息,历史由 checkpointer 按线程累积(spec「State 贯穿」);ch07:线程空
    而 DB 有史(重启/切换)→ ctx_store 回填近史前缀;store 经 configurable 供
    ctx/coref/agent 三消费面。
    ch08 T7:resume_value 非空=确认流续跑(以 Command(resume) 入图,不重建
    user 行);普通轮若检测到上轮 interrupt 未处置 → 先按 cancel drain(隐式取消,
    Review Focus 1);流尽后若仍挂 interrupt → 发 ("ticket_preview", {...}) 帧即关流。
    落库挂点:Agent 出口由 react 内部三挂点自落(ch04 同源);其余出口在此
    补 on_final_answer——user 行始终归 routes bootstrap。
    """
    graph = build_graph(settings, model)
    cfg = {"configurable": {"thread_id": None, "conversation_id": conversation_id,
                            "persister": persister, "ctx_store": ctx_store}}
    if conversation_id is not None:
        cfg["configurable"]["thread_id"] = f"conv-{conversation_id}"
    else:
        cfg["configurable"]["thread_id"] = f"anon-{uuid.uuid4()}"

    _tl = thread_lock(cfg["configurable"]["thread_id"]) if conversation_id is not None else None
    if resume_value is not None:
        # 终审 I-1:A 持锁跨过「查 pending(端点 dep)→resume 消费 interrupt→
        # 写 checkpoint」;首批帧 yield 前释放,B 的 drain 到锁时 pending 已灭。
        graph_input = Command(resume=resume_value)   # 确认流续跑,不重建输入
        if _tl is not None:
            await _tl.acquire()
            # 终审 I-1 反向:dep 检查与拿锁之间该 interrupt 可能已被新消息轮
            # drain 取消(客户秒发)——复查 pending,已灭则空续播(不重放节点、
            # 不双写取消话术入 messages,仅回显终局)。
            _snap_r = await graph.aget_state(cfg)
            if not (getattr(_snap_r, "interrupts", ()) or ()):
                graph_input = None
    else:
        current = chat_messages[-1].content
        # 隐式 cancel(卡片未处置就来新消息):先按取消 drain 旧单,本轮正常答,不 500。
        _drain_cm = _tl if _tl is not None else contextlib.nullcontext()
        async with _drain_cm:
            snap0 = await graph.aget_state(cfg)
            if getattr(snap0, "interrupts", ()):
                drained = await graph.ainvoke(Command(resume="cancel"), config=cfg)
                # M2-I2:取消说明必须落库——drain 无流帧,不落=DB 权威回填面(重启/
                # 切换)只见建单不见取消,历史回放缺终局。
                cancel_note = (drained or {}).get("answer_text") or ""
                if cancel_note and persister is not None:
                    try:
                        await persister.on_final_answer(conversation_id, cancel_note)
                    except Exception:  # noqa: BLE001 —— 落库失败只降级,不挡新轮
                        logger.warning("persist cancel note failed cid=%s",
                                       conversation_id, exc_info=True)
        input_msgs = [HumanMessage(content=current)]
        if conversation_id is not None and ctx_store is not None:
            # 终审 I2:回填=第四处 store 消费面,漏在 T7「store 面异常全吞」兜底
            # 清单外。DB 闪断等价 ch06 无 store 行为(无回填裸进),聊天不断线。
            try:
                prefix = await _refill_input(graph, cfg, ctx_store, settings, current)
            except Exception:  # noqa: BLE001
                logger.warning("refill degraded cid=%s", conversation_id, exc_info=True)
                prefix = []
            input_msgs = prefix + input_msgs
        elif conversation_id is None:
            input_msgs = to_langchain_messages(chat_messages)
        graph_input = {"messages": input_msgs, "user_query": current}

    streamed = False
    final: dict = {}
    # graph_input=None=resume 复查扑空(见上「反向」注)——不重放图,只走终态回显。
    _frames = (graph.astream(graph_input, config=cfg, stream_mode=["custom", "values"])
               if graph_input is not None else _empty_aiter())
    try:
        async for mode, chunk in _frames:
            if mode == "custom":
                kind = chunk[0]
                if kind == "token":
                    streamed = True
                if kind in ("token", "tool_call", "tool_result"):
                    yield chunk          # done 等内部事件不外发
            elif mode == "values":
                final = chunk
    finally:
        # 终审 I-1:A 的 checkpoint 已定稿(interrupt 已消耗)才交锁;循环内
        # 抛错/生成器被关也走此释放,不卡死该线程后续轮。
        if resume_value is not None and _tl is not None and _tl.locked():
            _tl.release()
    # 暂停轮检测:interrupt 挂起 → 发 preview 帧即关流(无 suggestions/落库补发)
    snap = await graph.aget_state(cfg)
    pend = getattr(snap, "interrupts", ()) or ()
    if pend:
        yield ("ticket_preview", {**dict(pend[0].value),
                                  "conversation_id": conversation_id})
        return
    answer = final.get("answer_text") or ""
    if answer and not streamed:      # 固定话术出口:无流式 token 时整段补发
        yield ("token", answer)
    # M2-I2:resume 轮 log.nodes 从 checkpoint 继承必含上轮 "agent",单判据
    # 「非 agent 出口」永不误触→T8 端点传入的 persister 成死线(工单号答复不落
    # messages,刷新即丢)。resume 轮图形=confirm→logging 恒不经 react 自落面,
    # 以 resume_value 为第二触发条件补发,无双写风险。
    if answer and (resume_value is not None
                   or "agent" not in (final.get("log") or {}).get("nodes", [])):
        try:
            if persister is not None:
                await persister.on_final_answer(conversation_id, answer)
        except Exception:  # noqa: BLE001 —— 落库失败只降级,不挡关流(spec §9)
            logger.warning("persist final answer failed (graph turn)", exc_info=True)
    if final.get("orders_payload"):  # ch06 P7:选择器卡片帧,仅非空才发(防御空帧)
        yield ("orders", {"items": final["orders_payload"]})
    if final.get("suggestions"):     # RF4:末 token 之后、done 之前
        yield ("suggestions", {"items": final["suggestions"]})
