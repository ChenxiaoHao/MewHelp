"""ReAct 主力 Agent 流式节点（ch05 spec「主力 Agent」节，需求 4）。

循环骨架 = Task 1 裸循环的流式版：bind_tools+astream 逐轮，文本实时透传
（ch04 同款 AIMessageChunk `+` 累加聚合），tool_calls 同轮逐个执行回喂
（R5 冒烟语义）。在裸循环之上加三件事：
1. 每轮 token 累计对 settings.react_token_budget 熔断（P3=8000）；
2. 轮数上限 settings.max_agent_steps（ch07 T8 接管，P3=6）；
3. 超限收尾不追加模型调用、外发最后一段模型自述 + suggestions=[转人工]
   （ledger R11：省一次调用与延迟，"已有信息收尾"的最小实现）。

ch07 T8 改形：证据/订单注入=装配段5（build_model_context）之责，本模块不再
读 state.evidence/order_data 做任何 System 前置；轮数上限=settings.
max_agent_steps（P3 接管）；token 计数与预算层同源（CJK 估算器 estimate_msg）。

事件元组与 ch04 ToolEvent 逐键对齐，done 为 T6 帧序新增；ticket_request
为 ch08 T7 内部事件（仅 agent_node 捕获转确认流，不进 writer 外发白名单）：
("token", str) | ("tool_call", {id,name,args})
| ("tool_result", {id,name,ok,summary[,citations]})
| ("ticket_request", {"tool_call_id", "args"})
| ("done", {"steps": int, "suggestions": list})
"""

import json
import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from app.context.budget import estimate_msg
from app.rag import confidence
from app.tools.executor import ToolContext, audit_denied, execute_tool, make_summary
from app.tools.registry import BUILTIN_SPECS
from app.workflows.state import TRANSFER_HUMAN

logger = logging.getLogger(__name__)

EXHAUST_PREFIX = "（已达最大轮数）"


def _serialize_tool_calls(tool_calls: list) -> list:
    """与 ch04 tool_chat_service 同形(落库 messages.tool_calls 列)。"""
    return [{"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args") or {}}
            for tc in tool_calls]


async def react_agent_stream(
    state: dict, settings: Any, model: Any, *, persister: Any = None,
    specs: dict | None = None,
) -> AsyncIterator[tuple]:
    """state 需含 messages；可选 evidence(知识路证据)、conversation_id(落库/超时上下文)。

    persister=ch04 ChatPersister 协议实现；三挂点调用位置与 stream_chat_with_tools
    一一对应（落库语义不动，P5），失败只 WARN 不中断流（spec §9）。
    """
    cid = state.get("conversation_id")
    # ch09 T5:知识轮快照随终答行落库(👎 回捞源);无证据=None(闲聊/数据面)。
    # M2-C-1:agent_node 恒带显式键(其保险丝会清 evidence,键是唯一可靠下传通道);
    # 键缺位=旧直调面(ch05/06 测形)契约不变——从 evidence 现算。
    if "retrieval_snapshot" in state:
        _snap = state["retrieval_snapshot"] or None
    else:
        _snap = confidence.evidence_snapshot(state.get("evidence"), settings) or None

    async def _p(hook: str, *args: Any, **kw: Any) -> None:
        if persister is None:
            return
        try:
            await getattr(persister, hook)(*args, **kw)
        except Exception:  # noqa: BLE001
            logger.warning("persister.%s failed; stream continues", hook, exc_info=True)

    messages: list = list(state["messages"])
    # ch07 T8:证据/订单不再 System 前置——段5 装配(build_model_context)已把
    # 「知识库证据」/「订单数据」合注入当前句之后一条 Human;此处读 state 会双份注入。
    # 退化面(无 store→legacy trim)见 Ruling:本轮 gate 仍拦弱证据,仅少文本注入。
    # ch08 T7:bind 快照全集(含 create_ticket,需求7)。留史:ch05 终审 F 批的 D2
    # 硬闸(绑定剔除+执行层拦截,原 :109-117)由确认流取代——模型可提案建单,
    # 权限闸(无凭证→awaiting_confirmation)拦下,客户在预览卡片确认后才真正写单。
    specs = specs if specs is not None else BUILTIN_SPECS
    bound = model.bind_tools([s.tool for s in specs.values()])
    ctx = ToolContext(
        conversation_id=state.get("conversation_id"),
        timeout_seconds=settings.tool_timeout_seconds,
        max_retries=settings.tool_max_retries,
    )
    steps = 0
    tokens_used = 0
    last_text = ""
    all_text: list[str] = []
    while steps < settings.max_agent_steps:
        full = None
        async for chunk in bound.astream(messages):
            full = chunk if full is None else full + chunk
            text = getattr(chunk, "text", "") or ""
            if text:
                tokens_used += estimate_msg(AIMessage(content=text))
                all_text.append(text)
                yield ("token", text)
        full = full if full is not None else AIMessage(content="")
        messages.append(full)
        if _text_of(full).strip():
            last_text = _text_of(full)
        tool_calls = list(getattr(full, "tool_calls", None) or [])
        if not tool_calls:
            await _p("on_final_answer", cid, _text_of(full),   # ch04 同位挂点
                     retrieval_snapshot=_snap)
            yield ("done", {"steps": steps, "suggestions": []})
            return
        if tokens_used >= settings.react_token_budget:          # 预算熔断:收尾即已发文本
            await _p("on_final_answer", cid, "".join(all_text),
                     retrieval_snapshot=_snap)
            yield ("done", {"steps": steps, "suggestions": [TRANSFER_HUMAN]})
            return
        await _p("on_tool_calls", cid, _text_of(full),
                 _serialize_tool_calls(tool_calls))              # ch04 同位挂点
        for tc in tool_calls:                                    # 同轮并行调用逐个执行(R5)
            yield ("tool_call", {"id": tc["id"], "name": tc["name"],
                                 "args": tc.get("args") or {}})
            # ch08 T7:ch05 D2 执行硬闸退役(git 史 556832f 前),create_ticket 与
            # 全体工具统一走查表+执行器——写闸拦截由 executor 权限闸承担。
            spec = specs.get(tc["name"])
            if spec is None:
                # 幻觉未登记调用=未授权:拒绝回灌 + 审计「权限拒绝」(Review Focus 6)
                await audit_denied(ctx, tc["name"], tc["id"], tc.get("args") or {})
                _err = f"未注册的工具: {tc['name']}"
                outcome = SimpleNamespace(
                    tool_call_id=tc["id"], name=tc["name"], ok=False,
                    summary=make_summary(tc["name"], {"error": _err}),
                    citations=None, result={"error": _err},
                )
            else:
                outcome = await execute_tool(spec, tc.get("args") or {}, tc["id"], ctx)
            if getattr(outcome, "awaiting_confirmation", False):
                # 需求7:写提案无凭证被闸拦下=中间态 → 转确认流事件(agent_node 捕获,
                # 不外发);tool_result 帧照发,模型见「等待客户确认」自收敛收尾。
                yield ("ticket_request", {"tool_call_id": tc["id"],
                                          "args": tc.get("args") or {}})
            payload = {"id": outcome.tool_call_id, "name": outcome.name,
                       "ok": outcome.ok, "summary": outcome.summary}
            if outcome.citations:                                # 与 ch04 逐字符同形(仅命中才加键)
                payload["citations"] = outcome.citations
            yield ("tool_result", payload)
            messages.append(ToolMessage(
                content=json.dumps(outcome.result, ensure_ascii=False, default=str),
                tool_call_id=tc["id"],
            ))
            await _p("on_tool_result", cid, outcome)             # ch04 同位挂点
        steps += 1
        logger.info("ch05 react step %d/%d tokens≈%d",
                    steps, settings.max_agent_steps, tokens_used)
    wrap = EXHAUST_PREFIX + last_text
    await _p("on_final_answer", cid, wrap, retrieval_snapshot=_snap)
    yield ("token", wrap)
    yield ("done", {"steps": steps, "suggestions": [TRANSFER_HUMAN]})


def _text_of(msg: Any) -> str:
    """AIMessage(Chunk).content 可能是 str 或分片 list（对齐 tool_chat_service 压法）。"""
    c = getattr(msg, "content", "") or ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)
