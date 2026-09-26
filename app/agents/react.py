"""ReAct 主力 Agent 流式节点（ch05 spec「主力 Agent」节，需求 4）。

循环骨架 = Task 1 裸循环的流式版：bind_tools+astream 逐轮，文本实时透传
（ch04 同款 AIMessageChunk `+` 累加聚合），tool_calls 同轮逐个执行回喂
（R5 冒烟语义）。在裸循环之上加三件事：
1. 每轮 token 累计对 settings.react_token_budget 熔断（P3=8000）；
2. 轮数上限 settings.react_max_iterations（P3=6）；
3. 超限收尾不追加模型调用、外发最后一段模型自述 + suggestions=[转人工]
   （ledger R11：省一次调用与延迟，"已有信息收尾"的最小实现）。

事件元组与 ch04 ToolEvent 逐键对齐，done 为 T6 帧序新增：
("token", str) | ("tool_call", {id,name,args})
| ("tool_result", {id,name,ok,summary[,citations]})
| ("done", {"steps": int, "suggestions": list})
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.tools.executor import ToolContext, execute_tool
from app.tools.registry import get_tools
from app.workflows.state import TRANSFER_HUMAN

logger = logging.getLogger(__name__)

EXHAUST_PREFIX = "（已达最大轮数）"


async def react_agent_stream(
    state: dict, settings: Any, model: Any
) -> AsyncIterator[tuple]:
    """state 需含 messages；可选 evidence(知识路证据)、conversation_id(落库/超时上下文)。"""
    messages: list = list(state["messages"])
    if state.get("evidence"):
        ev = "\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(state["evidence"]))
        messages = [SystemMessage(content=f"知识库证据:\n{ev}"), *messages]
    bound = model.bind_tools(get_tools())
    ctx = ToolContext(
        conversation_id=state.get("conversation_id"),
        timeout_seconds=settings.tool_timeout_seconds,
        max_retries=settings.tool_max_retries,
    )
    steps = 0
    tokens_used = 0
    last_text = ""
    while steps < settings.react_max_iterations:
        full = None
        async for chunk in bound.astream(messages):
            full = chunk if full is None else full + chunk
            text = getattr(chunk, "text", "") or ""
            if text:
                tokens_used += count_tokens_approximately(
                    [AIMessage(content=text)])
                yield ("token", text)
        full = full if full is not None else AIMessage(content="")
        messages.append(full)
        if _text_of(full).strip():
            last_text = _text_of(full)
        tool_calls = list(getattr(full, "tool_calls", None) or [])
        if not tool_calls:
            yield ("done", {"steps": steps, "suggestions": []})
            return
        if tokens_used >= settings.react_token_budget:      # 预算熔断:本轮已发文本即收尾
            yield ("done", {"steps": steps, "suggestions": [TRANSFER_HUMAN]})
            return
        for tc in tool_calls:                               # 同轮并行调用逐个执行(R5)
            yield ("tool_call", {"id": tc["id"], "name": tc["name"],
                                 "args": tc.get("args") or {}})
            outcome = await execute_tool(tc["name"], tc.get("args") or {}, tc["id"], ctx)
            payload = {"id": outcome.tool_call_id, "name": outcome.name,
                       "ok": outcome.ok, "summary": outcome.summary}
            if outcome.citations:                           # 与 ch04 逐字符同形(仅命中才加键)
                payload["citations"] = outcome.citations
            yield ("tool_result", payload)
            messages.append(ToolMessage(
                content=json.dumps(outcome.result, ensure_ascii=False, default=str),
                tool_call_id=tc["id"],
            ))
        steps += 1
        logger.info("ch05 react step %d/%d tokens≈%d",
                    steps, settings.react_max_iterations, tokens_used)
    yield ("token", EXHAUST_PREFIX + last_text)
    yield ("done", {"steps": steps, "suggestions": [TRANSFER_HUMAN]})


def _text_of(msg: Any) -> str:
    """AIMessage(Chunk).content 可能是 str 或分片 list（对齐 tool_chat_service 压法）。"""
    c = getattr(msg, "content", "") or ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)
