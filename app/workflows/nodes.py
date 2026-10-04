"""Graph 节点（ch05 spec「图拓扑」节）。

每个节点 = 一个闭包工厂,注入 settings/model,返回 `(state)->partial update`。
节点在 `log["nodes"]` 追加自身名,验收 1「日志可见强制检索节点被走到」即读这条链。
retrieve 经 `retriever.retrieve(...)` 模块属性调用点(非 from-import 绑定)——
测试据此 monkeypatch 注入假检索,生产代码零改动(D7 只读)。
"""

import asyncio
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from app.agents.react import react_agent_stream
from app.context.budget import compute_budgets, estimate_items
from app.core.config import get_settings
from app.context.layers import (
    _build_injection,
    build_history_view,
    build_model_context,
    degrade_if_needed,
    log_model_ctx,
    render_layer2,
)
from app.context.summarizer import schedule_summary
from app.prompts.coref import COREF_PROMPT
from app.prompts.customer_service import CUSTOMER_SERVICE_PROMPT
from app.prompts.intent import INTENT_PROMPT
from app.prompts.query_expand import EXPAND_PROMPT
from app.rag import confidence, retriever
from app.services import refusals          # 经模块属性调用,pool 可被测试替换
from app.services.chat_service import get_model, trim_history
from app.tools.definitions import _make_order, list_user_orders
from app.tools.executor import ToolContext, audit_denied, execute_tool
from app.tools.registry import BUILTIN_SPECS
from app.workflows.routing import (
    chitchat_fast_path,
    extract_order_id,
    match_order_selection,
    merge_evidence,
    parse_intent_json,
    parse_queries_json,
    route_for_intent,
)
from app.workflows.state import (
    CREATE_TICKET,
    REFUND_APPLY,
    SELECT_ORDER_ASK,
    TRANSFER_HUMAN,
)

logger = logging.getLogger(__name__)

# R1: 闲聊/投诉固定话术常量落在本模块(闸兜底复用 refusals.REFUSAL_ANSWER 不新造)。
CHITCHAT_FIXED = "您好，我是客服小猫，很高兴为您服务～有什么可以帮您的吗？"
COMPLAINT_FIXED = "非常抱歉给您带来了不好的体验，我们一定会认真处理。"


def _note(state: dict, name: str) -> dict:
    prev = dict(state.get("log") or {})
    prev["nodes"] = [*(prev.get("nodes") or []), name]
    return prev


def _store_of(config):
    return ((config or {}).get("configurable") or {}).get("ctx_store")


def make_ctx_node(settings, model):
    """ch07 需求 2:每轮 coref 之前的三层边界决策点(spec「后台异步摘要」)。

    有 store:读锚 → 层1 降级(本轮同步落库)→ 层2 渲染形态按 token 折判超预算
    → schedule_summary(后台异步,防重入交 T6)。无 store(cid=None/引擎降级面)
    零副作用;store 炸了只 WARN 不拖垮本轮——ctx 是加速器不是闸。
    """
    async def ctx_node(state: dict, config: RunnableConfig = None) -> dict:
        store = _store_of(config)
        if store is None:
            return {}
        try:
            await degrade_if_needed(store, settings)
            summary, upto, layer1_from = await store.load_ctx()
            rows = await store.fetch_all_rows()
            layer2 = [r for r in rows if upto < r.id <= layer1_from]
            if layer2:
                est = estimate_items(render_layer2(layer2, settings))
                budget = compute_budgets(settings).layer2
                if est > budget:
                    logger.info("summary trigger cid=%s 层2 约%d token > 预算%d",
                                store.cid, est, budget)
                    schedule_summary(store.session_factory, store.cid, settings, model)
        except Exception:  # noqa: BLE001
            logger.warning("ctx node degraded cid=%s", store.cid, exc_info=True)
        return {}
    return ctx_node


def _strip_current_turn(view: str, q: str) -> str:
    """剔掉 DB view 末行当前用户句(bootstrap 已先落 user 行)——自己不许当自己的历史。"""
    lines = view.splitlines()
    if lines and lines[-1].strip() == f"用户:{q}":
        lines = lines[:-1]
    return "\n".join(lines).strip() or "(无)"


def make_coref_node(model, settings=None):
    """ch06 正式版:LLM+历史把指代补全成自包含标准问法;已完整原样透传;首轮零调用。

    ch07 T7 换供:有 ctx_store 时 {history}=build_history_view(DB 权威,含梗概行,
    重启/空线程免疫),首轮判定仍看线程态(回填保证重启后线程非空→不误判首轮);
    history_ctx 每轮必打(含闲聊轮,本节点先于 intent/chitchat 执行即为「graph 级」)。

    兼作逐轮复位点(M1-I1 语义保留):轮级字段清零开新 log;ch06 新键一并复位,
    唯一例外 pending_flow 先读后清(Task 4 续跑识别在本函数顶端接管)。
    失败/空输出 → 透传原话,理解层绝不拖垮主流程(spec「失败与降级」)。
    """
    async def coref_node(state: dict, config: RunnableConfig = None) -> dict:
        q = state.get("user_query", "")
        store = _store_of(config)
        view = None
        if store is not None:
            # 每轮先构 DB 视图(顺带落 history_ctx,需求 6「每轮必打」含 resume/闲聊轮);
            # bootstrap 刚落的当前句从尾部剔掉——自己不许当自己的历史。
            try:
                view = _strip_current_turn(await build_history_view(store, settings), q)
            except Exception:  # noqa: BLE001 —— 理解层绝不拖垮主流程(spec「失败与降级」)
                logger.warning("coref history view failed cid=%s", store.cid, exc_info=True)
        # 续跑识别(T4,pending_flow 先读后清):选择器点卡片协议句 → 模板补全,零模型调用
        resume_oid = (match_order_selection(q)
                      if state.get("pending_flow") == "refund" else None)
        reset = {"log": {"nodes": ["coref"]}, "evidence": [], "suggestions": [],
                 "answer_text": "", "gate_pass": False, "pending_flow": "",
                 "slot_order_id": "", "order_data": {}, "expanded_queries": [],
                 "orders_payload": [], "intent_confidence": 0.0,
                 "ticket_preview": {}}   # ch08 T7:轮级复位,残卡不许跨轮路由
        if resume_oid:
            reset["log"].update(coref="resume", resume=True)
            return {**reset, "slot_order_id": resume_oid,
                    "resolved_query": f"订单 {resume_oid} 能不能申请退款？"}
        hist = list(state.get("messages") or [])[:-1]      # 本轮 human 已在图入参
        if not any(isinstance(m, HumanMessage) for m in hist):
            reset["log"]["coref"] = "passthrough"
            return {**reset, "resolved_query": q}
        history_text = view if view not in (None, "(无)") else _render_history(hist[-6:])
        try:
            msgs = COREF_PROMPT.format_messages(
                history=history_text, question=q)
            text = _text_of(await model.ainvoke(msgs)).strip()
            resolved = next((ln.strip() for ln in reversed(text.splitlines())
                             if ln.strip()), "")
        except Exception:  # noqa: BLE001
            reset["log"]["coref"] = "degraded"
            return {**reset, "resolved_query": q}
        # spec 降级表:LLM 异常与空输出同归 degraded(done 只留给真补全,M1-F1)
        reset["log"]["coref"] = "done" if resolved else "degraded"
        return {**reset, "resolved_query": resolved or q}
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
        if state.get("slot_order_id"):
            # 续跑直通(coref resume 已钉死办事意图):协议句无需再判类,零调用
            return {"intent": "退款退货", "route": "refund", "intent_confidence": 1.0,
                    "log": {**log, "intent": "退款退货", "confidence": 1.0,
                            "resume": True}}
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


# M1-I3 降级带:ch04 闸1 保证非降级非空证据 top1≥阈值(0.161),故低于 RRF 双路
# 理论上限(rrf_k=60 → ≈0.033)留裕度到 0.04 的分只可能来自重排降级路径——
# 对齐闸1「降级跳过」决定:该带内非空证据旁路过闸进 Agent,不落池。
# ch09 T4 判定核换 evidence_confidence 后此旁路语义原样保留(Review Focus 3)。
RRF_DEGRADED_MAX = 0.04


def _evidence_snapshot(evidence, settings) -> list[dict]:
    """快照形制收口 confidence.evidence_snapshot(T5 起 messages 行同源)。"""
    return confidence.evidence_snapshot(evidence, settings)


def make_confidence_gate_node(settings, source: str = "ch05_gate",
                              name: str = "gate"):
    async def confidence_gate_node(state: dict, config: RunnableConfig) -> dict:
        """ch09 T4 判定核:evidence_confidence(top1/n_eff/gap,校准定值)。

        位置与行为不变(spec 钉死):拦下 → 兜底话术+转人工建议+落池(带召回
        快照,reason=三信号复盘串)+fail 路由;source 参数化保留 ch06 分池归因;
        RRF 降级带旁路语义原样(过闸不落池,gate_scale="rrf_degraded");
        池写失败由 refusals 内部吞掉,不阻断兜底(Review Focus 5)。
        """
        evidence = state.get("evidence") or []
        verdict = confidence.evaluate(evidence, settings)
        scores = [e["score"] if isinstance(e, dict) else e.score for e in evidence]
        best = max(scores) if scores else 0.0
        degraded = (not verdict.ok) and bool(evidence) and best <= RRF_DEGRADED_MAX
        ok = verdict.ok or degraded
        log = _note(state, name)
        log["gate_pass"] = ok
        log["gate_best_score"] = round(best, 4)
        if degraded:
            log["gate_scale"] = "rrf_degraded"
        if not ok:
            query = state.get("resolved_query") or state.get("user_query", "")
            cid = (config.get("configurable") or {}).get("conversation_id")
            await refusals.pool_low_confidence(
                cid, query, source, verdict.detail,
                retrieved_chunks=_evidence_snapshot(evidence, settings))
            return {"gate_pass": False, "answer_text": refusals.REFUSAL_ANSWER,
                    "suggestions": [TRANSFER_HUMAN],
                    "messages": [AIMessage(content=refusals.REFUSAL_ANSWER)], "log": log}
        return {"gate_pass": True, "log": log}
    return confidence_gate_node


def make_agent_node(model, settings):
    async def agent_node(state: dict, config: RunnableConfig) -> dict:
        """Task 6 版:经 custom writer 直发 ReAct 事件帧,答案文本自 token 聚合。

        conversation_id/persister 均经 config.configurable 注入(R7 同款通道);
        进模型的消息列:有 store=五段装配(段5 已含证据/订单注入);退位=ch01
        人设渲染+预算裁剪,证据/订单由本节点尾挂段5 同形 Human(M3-I2)。
        非流式上下文(纯 ainvoke 测试)writer 不存在时静默降级,聚合语义不变。
        """
        conf = config.get("configurable") or {}
        cid = conf.get("conversation_id")
        try:
            writer = get_stream_writer()
        except RuntimeError:
            writer = lambda ev: None  # noqa: E731 —— 图外直调(理论不达)静默
        hist = list(state["messages"])
        store = _store_of(config)
        msgs = None
        if store is not None:
            # ch07 五段装配(DB 权威,含层2半压/层1原文/合并注入);装配面炸了
            # 退回 ch01 旧路径——聊天不断线优先(spec「预算退化」)。
            cur = next((m for m in reversed(hist) if isinstance(m, HumanMessage)),
                       HumanMessage(state.get("user_query", "")))
            try:
                msgs = await build_model_context(
                    store, evidence=state.get("evidence") or [],
                    order_data=state.get("order_data") or {},
                    current_human=cur, settings=settings)
                log_model_ctx(store.cid, msgs, estimate_items(msgs))
            except Exception:  # noqa: BLE001
                logger.warning("model_ctx build failed cid=%s; legacy assembly",
                               store.cid, exc_info=True)
                msgs = None
        five_seg = msgs is not None
        if msgs is None:
            msgs = trim_history(
                CUSTOMER_SERVICE_PROMPT.invoke({"messages": hist}).to_messages(),
                settings.history_token_budget,
            )
            # M3-I2:legacy 面 grounding 与 ch06 现状对齐(spec 范围红线「cid=None
            # 降级路径行为与现状一致」)——证据/订单并入段5 同一条 Human 尾挂;
            # System 旧形按 spec 段5 作废,形态可变、有无不可丢。
            inj = _build_injection(None, state.get("evidence") or [],
                                   state.get("order_data") or {}, settings)
            if inj is not None:
                msgs = [*msgs, inj]
        parts: list[str] = []
        done: dict = {"steps": 0, "suggestions": []}
        ticket_preview: dict | None = None
        react_input = {**state, "messages": msgs, "conversation_id": cid}
        if five_seg:
            # 段5 已把证据/订单合注入一条 Human(react System 旧路 T8 已废),
            # 清空入参防下游再读——防双份注入的保险丝。
            react_input.update(evidence=[], order_data={})
        # ch08 需求1/6:每轮现拿快照(内置 ∪ MCP),撞名内置存活、连不上只降级。
        from app.tools.registry import snapshot_tools
        specs = await snapshot_tools(settings)
        async for ev in react_agent_stream(
                react_input, settings, model, persister=conf.get("persister"),
                specs=specs):
            kind, data = ev
            if kind in ("token", "tool_call", "tool_result"):
                writer(ev)                       # ticket_request 不在白名单=不外发
            if kind == "token":
                parts.append(data)
            elif kind == "ticket_request":
                args = data.get("args") or {}
                ticket_preview = {"tool_call_id": data.get("tool_call_id"),
                                  "ticket_type": args.get("ticket_type", "咨询"),
                                  "description": args.get("description", "")}
            elif kind == "done":
                done = data
        answer = "".join(parts)
        log = _note(state, "agent")
        log["agent_steps"] = done["steps"]
        upd = {"answer_text": answer, "log": log,
               "messages": [AIMessage(content=answer)]}
        if ticket_preview:
            upd["ticket_preview"] = ticket_preview   # 交图条件边路由到 ticket_confirm
        if done["suggestions"]:
            upd["suggestions"] = done["suggestions"]   # 预算/轮数熔断转人工不被覆盖
        elif state.get("route") == "refund":
            # spec 追加拍板:闸过即固定挂退款入口(「不能退」时=仍要提交人工复核)
            upd["suggestions"] = [REFUND_APPLY]
        return upd
    return agent_node


# --- ch06 需求 5/6: 退款确定性子流程(槽位检→选择器|取单→扩写→政策检索→闸) -----


def refund_slot_node(state: dict) -> dict:
    """槽位检(纯同步,模型不许猜单号):已有 slot 或正文可正则提取;分支交条件边。"""
    oid = state.get("slot_order_id") or extract_order_id(
        state.get("resolved_query"), state.get("user_query"))
    log = _note(state, "refund_slot")
    return {"slot_order_id": oid or "", "log": log}


def make_refund_selector_node(settings):
    def refund_selector_node(state: dict) -> dict:
        """无单号 → 弹订单卡片(orders 帧,T5 接线)+ 挂 pending_flow 等续跑。"""
        log = _note(state, "refund_selector")
        return {"orders_payload": list_user_orders(settings.demo_user_id),
                "pending_flow": "refund", "answer_text": SELECT_ORDER_ASK,
                "messages": [AIMessage(content=SELECT_ORDER_ASK)], "log": log}
    return refund_selector_node


def refund_fetch_node(state: dict) -> dict:
    """取单:_make_order 与详情工具同播种(P4),交 Agent 注入用,不经模型选工具。"""
    log = _note(state, "refund_fetch")
    return {"order_data": _make_order(state["slot_order_id"]), "log": log}


def make_refund_expand_node(model):
    async def refund_expand_node(state: dict) -> dict:
        """Query 扩写(仅本高敏子流程;FAQ 不扩写):≤3 条侧重不同,原问法居首;
        烂输出/异常 → 单路 [resolved] 照常检索(降级不阻断)。"""
        log = _note(state, "refund_expand")
        resolved = state.get("resolved_query") or state.get("user_query", "")
        qs = None
        try:
            ai = await model.ainvoke(EXPAND_PROMPT.format_messages(question=resolved))
            qs = parse_queries_json(_text_of(ai))
        except Exception:  # noqa: BLE001
            logger.warning("refund expand degraded; single query", exc_info=True)
        expanded = [resolved] + [q for q in (qs or []) if q != resolved][:3]
        return {"expanded_queries": expanded, "log": log}
    return refund_expand_node


def make_refund_policy_node(settings):
    async def refund_policy_node(state: dict) -> dict:
        """多路政策检索(并行)→ chunk_id 去重取最高分 → 截 rerank_top_n(P3)。"""
        log = _note(state, "refund_policy")
        groups = await asyncio.gather(*[
            retriever.retrieve(q, strategy="hybrid_rerank", settings=settings)
            for q in state["expanded_queries"]])
        grps = [[{"chunk_id": c.chunk_id, "score": c.score,
                  "text": retriever.vector_text(c.row)} for c in g.chunks]
                for g in groups]
        merged = merge_evidence(grps, cap=settings.rerank_top_n)
        log["retrieve_hits"] = len(merged)
        return {"evidence": merged, "log": log}
    return refund_policy_node


def complaint_node(state: dict) -> dict:
    log = _note(state, "complaint")
    return {"answer_text": COMPLAINT_FIXED,
            "suggestions": [TRANSFER_HUMAN, CREATE_TICKET],
            "messages": [AIMessage(content=COMPLAINT_FIXED)], "log": log}


def chitchat_node(state: dict) -> dict:
    log = _note(state, "chitchat")
    return {"answer_text": CHITCHAT_FIXED,
            "messages": [AIMessage(content=CHITCHAT_FIXED)], "log": log}


async def ticket_confirm_node(state: dict, config: RunnableConfig) -> dict:
    """ch08 确认节点(需求7,spec 确认流节):interrupt 前置零副作用——
    langgraph 恢复语义=本节点从头重放(T7 探针实证 count 1→2),
    execute/审计只在 interrupt() 返回之后(P6)。"""
    preview = state.get("ticket_preview") or {}
    decision = interrupt(dict(preview))          # 第一行动:此前不得有任何副作用
    conf = (config or {}).get("configurable") or {}
    cid = conf.get("conversation_id")
    log = _note(state, "ticket_confirm")
    args = {"description": preview.get("description", ""),
            "ticket_type": preview.get("ticket_type", "咨询")}
    if decision == "confirm":
        ctx = ToolContext(conversation_id=cid,
                          timeout_seconds=get_settings().tool_timeout_seconds,
                          ticket_confirmed=True)
        outcome = await execute_tool(BUILTIN_SPECS["create_ticket"], args,
                                     preview.get("tool_call_id") or "confirm", ctx)
        ans = (f"已为您创建工单 {outcome.result['ticket_no']}，处理进度会另行通知。"
               if outcome.ok else
               f"工单创建失败：{outcome.summary}，请稍后再试或使用页面下方「建工单」按钮。")
    else:
        # 终局取消才落账(T4 裁决延伸):中间态拒绝不发审计,这条=权限拒绝终局
        await audit_denied(ToolContext(conversation_id=cid), "create_ticket",
                           preview.get("tool_call_id") or "cancel", args,
                           reason="客户在预览卡片取消，未执行")
        ans = "好的，已取消本次建单。"
    return {"answer_text": ans, "log": log,
            "messages": [AIMessage(content=ans)], "ticket_preview": {}}


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
