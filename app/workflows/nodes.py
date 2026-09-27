"""Graph 节点（ch05 spec「图拓扑」节）。

每个节点 = 一个闭包工厂,注入 settings/model,返回 `(state)->partial update`。
节点在 `log["nodes"]` 追加自身名,验收 1「日志可见强制检索节点被走到」即读这条链。
retrieve 经 `retriever.retrieve(...)` 模块属性调用点(非 from-import 绑定)——
测试据此 monkeypatch 注入假检索,生产代码零改动(D7 只读)。
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from app.agents.react import react_agent_stream
from app.prompts.coref import COREF_PROMPT
from app.prompts.customer_service import CUSTOMER_SERVICE_PROMPT
from app.prompts.intent import INTENT_PROMPT
from app.rag import retriever
from app.services import refusals          # 经模块属性调用,pool 可被测试替换
from app.services.chat_service import get_model, trim_history
from app.workflows.routing import (
    chitchat_fast_path,
    parse_intent_json,
    route_for_intent,
)
from app.workflows.state import CREATE_TICKET, TRANSFER_HUMAN

logger = logging.getLogger(__name__)

# R1: 闲聊/投诉固定话术常量落在本模块(闸兜底复用 refusals.REFUSAL_ANSWER 不新造)。
CHITCHAT_FIXED = "您好，我是客服小猫，很高兴为您服务～有什么可以帮您的吗？"
COMPLAINT_FIXED = "非常抱歉给您带来了不好的体验，我们一定会认真处理。"


def _note(state: dict, name: str) -> dict:
    prev = dict(state.get("log") or {})
    prev["nodes"] = [*(prev.get("nodes") or []), name]
    return prev


def make_coref_node(model):
    """ch06 正式版:LLM+历史把指代补全成自包含标准问法;已完整原样透传;首轮零调用。

    兼作逐轮复位点(M1-I1 语义保留):轮级字段清零开新 log;ch06 新键一并复位,
    唯一例外 pending_flow 先读后清(Task 4 续跑识别在本函数顶端接管)。
    失败/空输出 → 透传原话,理解层绝不拖垮主流程(spec「失败与降级」)。
    """
    async def coref_node(state: dict) -> dict:
        q = state.get("user_query", "")
        reset = {"log": {"nodes": ["coref"]}, "evidence": [], "suggestions": [],
                 "answer_text": "", "gate_pass": False, "pending_flow": "",
                 "slot_order_id": "", "order_data": {}, "expanded_queries": [],
                 "orders_payload": [], "intent_confidence": 0.0}
        hist = list(state.get("messages") or [])[:-1]      # 本轮 human 已在图入参
        if not any(isinstance(m, HumanMessage) for m in hist):
            reset["log"]["coref"] = "passthrough"
            return {**reset, "resolved_query": q}
        try:
            msgs = COREF_PROMPT.format_messages(
                history=_render_history(hist[-6:]), question=q)
            text = _text_of(await model.ainvoke(msgs)).strip()
            resolved = next((ln.strip() for ln in reversed(text.splitlines())
                             if ln.strip()), "") or q
        except Exception:  # noqa: BLE001
            reset["log"]["coref"] = "degraded"
            return {**reset, "resolved_query": q}
        reset["log"]["coref"] = "done"
        return {**reset, "resolved_query": resolved}
    return coref_node


def _render_history(msgs) -> str:
    lines = []
    for m in msgs:
        who = "用户" if isinstance(m, HumanMessage) else "客服"
        t = _text_of(m).strip()
        if t:
            lines.append(f"{who}:{t}")
    return "\n".join(lines) or "(无)"


async def _judge(m, query: str) -> tuple[str, float] | None:
    """一次判类:prompt→ainvoke→parse;调用异常与不合 schema 同归 None(不抛穿)。"""
    try:
        ai = await m.ainvoke(INTENT_PROMPT.format_messages(question=query))
        return parse_intent_json(_text_of(ai))
    except Exception:  # noqa: BLE001 —— 降级路上小模型挂了=等价未配置
        logger.warning("intent judge failed; escalate/fallback", exc_info=True)
        return None


async def _judge_retry(m, query: str) -> tuple[str, float] | None:
    for _ in range(2):                                      # ch05 P2 重试语义保留
        verdict = await _judge(m, query)
        if verdict:
            return verdict
    return None


def make_intent_node(model, settings):
    """ch06 四件套节点:快路(原话匹配)→ 降级路小模型先判/低置信升大模型 →
    解析失败×2 归「其他」(route=knowledge,替掉 ch05 归商品咨询的兜底,P1)。"""
    async def intent_node(state: dict) -> dict:
        log = _note(state, "intent")
        query = state.get("resolved_query") or state.get("user_query", "")
        if chitchat_fast_path(state.get("user_query", "")):  # D5 + ch06:快路看原话
            return {"intent": "闲聊", "route": "chitchat", "intent_confidence": 1.0,
                    "log": {**log, "fast_path": True, "intent": "闲聊"}}
        verdict = None
        if getattr(settings, "intent_small_model", ""):      # P2:默认空=只走大模型
            verdict = await _judge(
                get_model(settings, model_name=settings.intent_small_model), query)
            if verdict and verdict[1] < settings.intent_confidence_threshold:
                verdict = None                               # 低置信 → 大模型复判
        if verdict is None:
            verdict = await _judge_retry(model, query)
        intent, conf = verdict or ("其他", 0.0)
        return {"intent": intent, "route": route_for_intent(intent),
                "intent_confidence": conf,
                "log": {**log, "intent": intent, "confidence": conf}}
    return intent_node


def make_knowledge_retrieve_node(settings):
    async def knowledge_retrieve_node(state: dict) -> dict:
        res = await retriever.retrieve(
            state["resolved_query"], strategy="hybrid_rerank", settings=settings
        )
        evidence = [{"chunk_id": c.chunk_id, "score": c.score,
                     "text": retriever.vector_text(c.row)} for c in res.chunks]
        log = _note(state, "retrieve")
        log["retrieve_hits"] = len(evidence)
        return {"evidence": evidence, "log": log}
    return knowledge_retrieve_node


def evidence_gate_verdict(items, threshold: float) -> tuple[bool, float]:
    """纯判定:evidence(score 取自 dict 或 ScoredRow)最高分是否达阈值。

    阈值复用 ch04 闸1 终值 settings.retrieval_low_conf_threshold(P1,不新造键)。
    """
    scores = [i["score"] if isinstance(i, dict) else i.score for i in items]
    best = max(scores) if scores else 0.0
    return bool(scores) and best >= threshold, best


# M1-I3 降级带:ch04 闸1 保证非降级非空证据 top1≥阈值(0.161),故低于 RRF 双路
# 理论上限(rrf_k=60 → ≈0.033)留裕度到 0.04 的分只可能来自重排降级路径——
# 对齐闸1「降级跳过」决定:该带内非空证据旁路过闸进 Agent,不落池。
RRF_DEGRADED_MAX = 0.04


def make_confidence_gate_node(settings):
    async def confidence_gate_node(state: dict, config: RunnableConfig) -> dict:
        """Task 4 阈值版(P1):弱证据 → 兜底话术 + 落低置信池(source=ch05_gate)。

        池写失败由 refusals.pool_low_confidence 内部吞掉,不阻断兜底(Review Focus 5)。
        """
        evidence = state.get("evidence") or []
        ok, best = evidence_gate_verdict(evidence, settings.retrieval_low_conf_threshold)
        degraded = (not ok) and bool(evidence) and best <= RRF_DEGRADED_MAX
        if degraded:
            ok = True
        log = _note(state, "gate")
        log["gate_pass"] = ok
        log["gate_best_score"] = round(best, 4)
        if degraded:
            log["gate_scale"] = "rrf_degraded"
        if not ok:
            query = state.get("resolved_query") or state.get("user_query", "")
            cid = (config.get("configurable") or {}).get("conversation_id")
            await refusals.pool_low_confidence(
                cid, query, "ch05_gate",
                f"best={best:.4f}<{settings.retrieval_low_conf_threshold}")
            return {"gate_pass": False, "answer_text": refusals.REFUSAL_ANSWER,
                    "suggestions": [TRANSFER_HUMAN],
                    "messages": [AIMessage(content=refusals.REFUSAL_ANSWER)], "log": log}
        return {"gate_pass": True, "log": log}
    return confidence_gate_node


def make_agent_node(model, settings):
    async def agent_node(state: dict, config: RunnableConfig) -> dict:
        """Task 6 版:经 custom writer 直发 ReAct 事件帧,答案文本自 token 聚合。

        conversation_id/persister 均经 config.configurable 注入(R7 同款通道);
        进模型的消息列 = ch01 人设 Prompt 渲染 + 预算裁剪(ch04 同参),再由
        react 前置证据 SystemMessage。非流式上下文(纯 ainvoke 测试)writer
        不存在时静默降级,聚合语义不变。
        """
        conf = config.get("configurable") or {}
        cid = conf.get("conversation_id")
        try:
            writer = get_stream_writer()
        except RuntimeError:
            writer = lambda ev: None  # noqa: E731 —— 图外直调(理论不达)静默
        hist = list(state["messages"])
        msgs = trim_history(
            CUSTOMER_SERVICE_PROMPT.invoke({"messages": hist}).to_messages(),
            settings.history_token_budget,
        )
        parts: list[str] = []
        done: dict = {"steps": 0, "suggestions": []}
        async for ev in react_agent_stream(
                {**state, "messages": msgs, "conversation_id": cid},
                settings, model, persister=conf.get("persister")):
            kind, data = ev
            if kind in ("token", "tool_call", "tool_result"):
                writer(ev)
            if kind == "token":
                parts.append(data)
            elif kind == "done":
                done = data
        answer = "".join(parts)
        log = _note(state, "agent")
        log["agent_steps"] = done["steps"]
        upd = {"answer_text": answer, "log": log,
               "messages": [AIMessage(content=answer)]}
        if done["suggestions"]:
            upd["suggestions"] = done["suggestions"]
        return upd
    return agent_node


def complaint_node(state: dict) -> dict:
    log = _note(state, "complaint")
    return {"answer_text": COMPLAINT_FIXED,
            "suggestions": [TRANSFER_HUMAN, CREATE_TICKET],
            "messages": [AIMessage(content=COMPLAINT_FIXED)], "log": log}


def chitchat_node(state: dict) -> dict:
    log = _note(state, "chitchat")
    return {"answer_text": CHITCHAT_FIXED,
            "messages": [AIMessage(content=CHITCHAT_FIXED)], "log": log}


def logging_node(state: dict) -> dict:
    log = _note(state, "logging")
    # 前缀 "ch05 graph turn" 字面保留:双章 e2e 的锚点(改名无行为收益,记账 R 批)。
    line = {k: log.get(k) for k in ("nodes", "intent", "confidence", "gate_pass",
                                    "retrieve_hits", "agent_steps")}
    line["resolved"] = (state.get("resolved_query") or "")[:80]   # B1 验收断言面
    logger.info("ch05 graph turn %s", line)
    return {"log": log}


def _text_of(msg) -> str:
    c = getattr(msg, "content", "") or ""
    return c if isinstance(c, str) else "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in c
    )
