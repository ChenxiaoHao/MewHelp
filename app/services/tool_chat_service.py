"""单轮 Function Calling 编排（spec §5）。

流程：build_messages（ch01 复用）→ 第一轮 bind_tools+astream（文本实时透传，
chunk 用 + 累加聚合 tool_calls，写法经硬性核对点①实测）→ 有工具申请则逐个
执行并 yield 状态帧 → 结果以 ToolMessage 回灌 → 第二轮**裸 model**（不
bind_tools，物理保证单轮收敛）流式产出最终回答。

ch01 兼容：纯闲聊路径的事件流与 chat_service.stream_chat 完全等价；
本文件不修改 chat_service 的任何代码。
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol

from langchain_core.messages import AIMessageChunk, ToolMessage

from app.core.config import Settings
from app.schemas.chat import ChatMessage
from app.services.chat_service import build_messages, stream_chat
from app.tools.executor import ToolContext, ToolOutcome, execute_tool
from app.tools.registry import get_tools

logger = logging.getLogger(__name__)

# ("token", str) | ("tool_call", {id,name,args}) | ("tool_result", {id,name,ok,summary})
ToolEvent = tuple[str, Any]


class ChatPersister(Protocol):
    """编排层落库挂点（spec §7）。on_turn_start 在路由层直接走 crud。"""

    async def on_tool_calls(self, conversation_id: int, content: str, tool_calls: list) -> None: ...

    async def on_tool_result(self, conversation_id: int, outcome: ToolOutcome) -> None: ...

    async def on_final_answer(self, conversation_id: int, content: str) -> None: ...


def _text_of(msg: Any) -> str:
    """AIMessage(Chunk).content 可能是 str 或分片 list，统一压成 str。"""
    c = getattr(msg, "content", "") or ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in c
        )
    return str(c)


def _serialize_tool_calls(tool_calls: list) -> list:
    """tool_calls → 可 JSON 化 list（落库 messages.tool_calls 列）。"""
    return [
        {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args") or {}}
        for tc in tool_calls
    ]


async def _persist(persister: ChatPersister | None, hook: str, *args: Any) -> None:
    """落库失败只 warning，绝不打断 SSE（spec §7/§9）。"""
    if persister is None:
        return
    try:
        await getattr(persister, hook)(*args)
    except Exception:  # noqa: BLE001
        logger.warning("persister.%s failed; stream continues", hook, exc_info=True)


async def stream_chat_with_tools(
    chat_messages: list[ChatMessage],
    settings: Settings,
    model: Any,
    *,
    conversation_id: int | None = None,
    persister: ChatPersister | None = None,
) -> AsyncIterator[ToolEvent]:
    messages = build_messages(chat_messages, settings)
    bound = model.bind_tools(get_tools())
    ctx = ToolContext(
        conversation_id=conversation_id,
        timeout_seconds=settings.tool_timeout_seconds,
        max_retries=settings.tool_max_retries,
    )

    # ---- 第一轮：bind_tools + astream，文本实时透传，chunk 累加聚合 ----
    full: AIMessageChunk | None = None
    async for chunk in bound.astream(messages):
        full = chunk if full is None else full + chunk
        text = getattr(chunk, "text", "") or ""
        if text:
            yield ("token", text)

    tool_calls = list(getattr(full, "tool_calls", None) or []) if full is not None else []
    if not tool_calls:
        # ---- 纯闲聊：第一轮即最终回答（与 ch01 行为等价）----
        await _persist(persister, "on_final_answer", conversation_id, _text_of(full) if full else "")
        return

    # ---- 有工具决策：先落 assistant(tool_calls) 行，再逐个执行 ----
    await _persist(
        persister, "on_tool_calls", conversation_id,
        _text_of(full), _serialize_tool_calls(tool_calls),
    )

    round2: list = [*messages, full]  # 第一轮完整 AIMessage（含 tool_calls）回传上游
    for tc in tool_calls:
        yield ("tool_call", {"id": tc["id"], "name": tc["name"], "args": tc.get("args") or {}})
        outcome = await execute_tool(tc["name"], tc.get("args") or {}, tc["id"], ctx)
        payload = {
            "id": outcome.tool_call_id,
            "name": outcome.name,
            "ok": outcome.ok,
            "summary": outcome.summary,
        }
        if outcome.citations:  # 仅 query_faq 且有命中时加键,其余帧与旧结构逐字符一致
            payload["citations"] = outcome.citations
        yield ("tool_result", payload)
        round2.append(
            ToolMessage(
                content=json.dumps(outcome.result, ensure_ascii=False, default=str),
                tool_call_id=outcome.tool_call_id,
            )
        )
        await _persist(persister, "on_tool_result", conversation_id, outcome)

    # ---- 第二轮：裸 model（不 bind_tools）+ ch01 stream_chat，物理保证单轮收敛 ----
    parts: list[str] = []
    async for t in stream_chat(round2, model):
        parts.append(t)
        yield ("token", t)
    await _persist(persister, "on_final_answer", conversation_id, "".join(parts))
