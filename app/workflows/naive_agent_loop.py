"""祛魅热身：不借框架，手写的裸 Agent 循环（ch05 spec「总体架构」节·裸循环对照）。

看清一件事——所谓 Agent 就是「带工具的 while 循环」：
调 LLM → 有 tool_calls 就逐个执行、结果以 ToolMessage 喂回 → 没有就收敛出答案。
本模块是教学对照与 Task 5 ReAct 节点的语义基准，不接聊天入口（生产路径是 Graph）。
冒烟实测（dev-notes 2026-09-26）：上游模型会在一轮内并行返回多个 tool_calls，
循环必须逐轮全量执行，不能只取第一个。
"""

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import ToolMessage

from app.tools.executor import ToolContext, ToolOutcome, execute_tool, make_summary
from app.tools.registry import BUILTIN_SPECS, get_tools


@dataclass
class NaiveLoopResult:
    text: str
    steps: int
    tool_calls: list[ToolOutcome] = field(default_factory=list)


def _text_of(msg: Any) -> str:
    """AIMessage.content 可能是 str 或分片 list（对齐 tool_chat_service 的压法）。"""
    c = getattr(msg, "content", "") or ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


async def naive_agent_turn(
    model: Any,
    messages: list,
    *,
    ctx: ToolContext,
    max_iters: int = 6,
) -> NaiveLoopResult:
    """就地扩展传入的 messages（教学版刻意可见「历史在长大」）。"""
    bound = model.bind_tools(get_tools())
    outcomes: list[ToolOutcome] = []
    steps = 0
    last_text = ""
    for _ in range(max_iters):
        ai = await bound.ainvoke(messages)
        messages.append(ai)
        if _text_of(ai).strip():          # M1-I2:收尾用最后一段模型自述,不用工具 JSON
            last_text = _text_of(ai)
        tool_calls = list(getattr(ai, "tool_calls", None) or [])
        if not tool_calls:
            return NaiveLoopResult(text=_text_of(ai), steps=steps, tool_calls=outcomes)
        for tc in tool_calls:  # 同轮并行调用逐个执行（冒烟实测语义）
            spec = BUILTIN_SPECS.get(tc["name"])
            if spec is None:
                _err = f"未注册的工具: {tc['name']}"
                outcome = ToolOutcome(tc["name"], tc["id"], False, {"error": _err},
                                      make_summary(tc["name"], {"error": _err}))
            else:
                outcome = await execute_tool(spec, tc.get("args") or {}, tc["id"], ctx)
            outcomes.append(outcome)
            messages.append(
                ToolMessage(
                    content=json.dumps(outcome.result, ensure_ascii=False, default=str),
                    tool_call_id=tc["id"],
                )
            )
        steps += 1
    return NaiveLoopResult(
        text="（已达最大轮数）" + last_text,
        steps=steps,
        tool_calls=outcomes,
    )
