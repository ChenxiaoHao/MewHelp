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

事件元组与 ch04 ToolEvent 逐键对齐，done 为 T6 帧序新增：
("token", str) | ("tool_call", {id,name,args})
| ("tool_result", {id,name,ok,summary[,citations]})
| ("done", {"steps": int, "suggestions": list})
"""

import json
import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from app.context.budget import estimate_msg
from app.tools.executor import ToolContext, audit_denied, execute_tool, make_summary
from app.tools.registry import BUILTIN_SPECS, get_tools
from app.workflows.state import TRANSFER_HUMAN

logger = logging.getLogger(__name__)

EXHAUST_PREFIX = "（已达最大轮数）"


def _serialize_tool_calls(tool_calls: list) -> list:
    """与 ch04 tool_chat_service 同形(落库 messages.tool_calls 列)。"""
    return [{"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args") or {}}
            for tc in tool_calls]


async def react_agent_stream(
    state: dict, settings: Any, model: Any, *, persister: Any = None
) -> AsyncIterator[tuple]:
    """state 需含 messages；可选 evidence(知识路证据)、conversation_id(落库/超时上下文)。

    persister=ch04 ChatPersister 协议实现；三挂点调用位置与 stream_chat_with_tools
    一一对应（落库语义不动，P5），失败只 WARN 不中断流（spec §9）。
    """
    cid = state.get("conversation_id")

    async def _p(hook: str, *args: Any) -> None:
        if persister is None:
            return
        try:
            await getattr(persister, hook)(*args)
        except Exception:  # noqa: BLE001
            logger.warning("persister.%s failed; stream continues", hook, exc_info=True)

    messages: list = list(state["messages"])
    # ch07 T8:证据/订单不再 System 前置——段5 装配(build_model_context)已把
    # 「知识库证据」/「订单数据」合注入当前句之后一条 Human;此处读 state 会双份注入。
    # 退化面(无 store→legacy trim)见 Ruling:本轮 gate 仍拦弱证据,仅少文本注入。
    # D2「Agent 自动建单方案作废」+ 需求 8 红线:建单唯一入口=前端按钮→
    # POST /api/tickets。绑定集剔除 create_ticket(registry 保留件供执行器层复用,
    # 模型侧不可自触;终审修复批)。
    bound = model.bind_tools(
        [t for t in get_tools() if t.name != "create_ticket"]
    )
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
            await _p("on_final_answer", cid, _text_of(full))   # ch04 同位挂点
            yield ("done", {"steps": steps, "suggestions": []})
            return
        if tokens_used >= settings.react_token_budget:          # 预算熔断:收尾即已发文本
            await _p("on_final_answer", cid, "".join(all_text))
            yield ("done", {"steps": steps, "suggestions": [TRANSFER_HUMAN]})
            return
        await _p("on_tool_calls", cid, _text_of(full),
                 _serialize_tool_calls(tool_calls))              # ch04 同位挂点
        for tc in tool_calls:                                    # 同轮并行调用逐个执行(R5)
            yield ("tool_call", {"id": tc["id"], "name": tc["name"],
                                 "args": tc.get("args") or {}})
            if tc["name"] == "create_ticket":
                # 执行闸(D2/需求 8):建单唯一入口=前端按钮→POST /api/tickets;
                # 模型幻调即便过了 bind_tools 也绝不让触到执行器,回 ok=False。
                outcome = SimpleNamespace(
                    tool_call_id=tc["id"], name=tc["name"], ok=False,
                    summary="建工单仅可由页面「建工单」按钮触发，模型不可自调",
                    citations=None,
                    result={"error": "create_ticket is frontend-button-only"},
                )
            else:
                spec = BUILTIN_SPECS.get(tc["name"])
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
    await _p("on_final_answer", cid, wrap)
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
