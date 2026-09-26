"""Graph 节点（ch05 spec「图拓扑」节）。

每个节点 = 一个闭包工厂,注入 settings/model,返回 `(state)->partial update`。
节点在 `log["nodes"]` 追加自身名,验收 1「日志可见强制检索节点被走到」即读这条链。
retrieve 经 `retriever.retrieve(...)` 模块属性调用点(非 from-import 绑定)——
测试据此 monkeypatch 注入假检索,生产代码零改动(D7 只读)。
"""

import logging

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.prompts.intent import INTENT_PROMPT
from app.rag import retriever
from app.services.refusals import REFUSAL_ANSWER
from app.tools.executor import ToolContext
from app.workflows.naive_agent_loop import naive_agent_turn
from app.workflows.routing import (
    chitchat_fast_path,
    parse_intent_json,
    route_for_intent,
)

logger = logging.getLogger(__name__)

# R1: 闲聊/投诉固定话术常量落在本模块(闸兜底复用 refusals.REFUSAL_ANSWER 不新造)。
CHITCHAT_FIXED = "您好，我是客服小猫，很高兴为您服务～有什么可以帮您的吗？"
COMPLAINT_FIXED = "非常抱歉给您带来了不好的体验，我们一定会认真处理。"

TRANSFER_HUMAN = {"action": "transfer_human", "label": "转人工"}
CREATE_TICKET = {"action": "create_ticket", "label": "建工单"}


def _note(state: dict, name: str) -> dict:
    prev = dict(state.get("log") or {})
    prev["nodes"] = [*(prev.get("nodes") or []), name]
    return prev


def coref_node(state: dict) -> dict:
    """指代消解:本章原样透传(D1/需求 6,正式版留下一章)。"""
    log = _note(state, "coref")
    return {"resolved_query": state.get("user_query", ""), "log": log}


def make_intent_node(model):
    async def intent_node(state: dict) -> dict:
        log = _note(state, "intent")
        query = state.get("resolved_query") or state.get("user_query", "")
        if chitchat_fast_path(query):                       # D5:寒暄零模型调用
            return {"intent": "闲聊", "route": "chitchat",
                    "log": {**log, "fast_path": True}}
        intent = None
        for _ in range(2):                                  # P2:解析失败重试一次
            msgs = INTENT_PROMPT.format_messages(question=query)
            ai = await model.ainvoke(msgs)
            intent = parse_intent_json(_text_of(ai))
            if intent:
                break
        if intent is None:
            intent = "商品咨询"                             # 兜底归 knowledge(P2)
        return {"intent": intent, "route": route_for_intent(intent),
                "log": {**log, "intent": intent}}
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


def make_confidence_gate_node(settings):
    async def confidence_gate_node(state: dict) -> dict:
        """Task 3 直答版:非空即过;Task 4 换阈值版(签名/拓扑不变)。"""
        passed = bool(state.get("evidence"))
        log = _note(state, "gate")
        log["gate_pass"] = passed
        if not passed:
            return {"gate_pass": False, "answer_text": REFUSAL_ANSWER,
                    "messages": [AIMessage(content=REFUSAL_ANSWER)], "log": log}
        return {"gate_pass": True, "log": log}
    return confidence_gate_node


def make_agent_node(model, settings):
    async def agent_node(state: dict, config: RunnableConfig) -> dict:
        messages = list(state["messages"])
        if state.get("evidence"):
            ev = "\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(state["evidence"]))
            messages = [SystemMessage(content=f"知识库证据:\n{ev}"), *messages]
        ctx = ToolContext(
            conversation_id=(config.get("configurable") or {}).get("conversation_id"),
            timeout_seconds=settings.tool_timeout_seconds,
            max_retries=settings.tool_max_retries,
        )
        res = await naive_agent_turn(model, messages, ctx=ctx)   # Task 5 换流式版
        log = _note(state, "agent")
        log["agent_steps"] = res.steps
        return {"answer_text": res.text, "log": log,
                "messages": [AIMessage(content=res.text)]}
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
    logger.info("ch05 graph turn %s", {k: log.get(k) for k in
                ("nodes", "intent", "gate_pass", "retrieve_hits", "agent_steps")})
    return {"log": log}


def _text_of(msg) -> str:
    c = getattr(msg, "content", "") or ""
    return c if isinstance(c, str) else "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in c
    )
