"""工具执行基础设施：查找注册表、Schema 由 LangChain @tool 声明式校验、
asyncio.wait_for 超时、有限重试、异常→错误结果包装、摘要生成。

失败兜底原则（spec §9）：execute_tool 绝不向上抛——任何失败都变成
ok=False 的 ToolOutcome，其 result["error"] 作为 ToolMessage 回灌给模型，
让模型向用户礼貌收敛，而不是打断 SSE 流。
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from app.rag.hit_format import build_citations
from app.tools.registry import get_tool

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    conversation_id: int | None = None
    timeout_seconds: float = 5.0
    max_retries: int = 1


@dataclass
class ToolOutcome:
    name: str
    tool_call_id: str
    ok: bool
    result: Any  # dict：成功=工具返回；失败={"error": "..."}
    summary: str  # ≤80 字符：前端徽章文案 / tool 消息落库 content
    citations: list | None = None  # ch04: query_faq 命中集引用(前端弹窗数据);其余工具恒 None


def make_summary(name: str, result: Any) -> str:
    """把工具结果压成一句 ≤80 字符的中文摘要（前端徽章/落库用）。"""
    if not isinstance(result, dict):
        text = str(result)
    elif "error" in result:
        text = f"失败: {result['error']}"
    elif name == "query_faq":
        hits = result.get("hits", [])
        text = "未命中" if not hits else f"命中 {len(hits)} 条"
    elif name == "create_ticket":
        text = f"工单 {result.get('ticket_no', '?')} 已创建"
    elif name in ("query_order", "query_product"):
        status = result.get("current_status") or result.get("status")
        label = result.get("name") or result.get("carrier") or ""
        text = " ".join(x for x in (status, label) if x) or json.dumps(result, ensure_ascii=False)
    else:
        text = json.dumps(result, ensure_ascii=False)
    return text[:80]


async def execute_tool(
    name: str, args: dict, tool_call_id: str, context: ToolContext
) -> ToolOutcome:
    tool = get_tool(name)
    if tool is None:
        err = f"未注册的工具: {name}"
        logger.warning("tool lookup failed: %s", name)
        return ToolOutcome(name, tool_call_id, False, {"error": err}, make_summary(name, {"error": err}))

    config = {"configurable": {"conversation_id": context.conversation_id}}
    attempts = max(1, context.max_retries + 1)
    last_err = "未知错误"
    for i in range(attempts):
        try:
            result = await asyncio.wait_for(
                tool.ainvoke(args, config=config), timeout=context.timeout_seconds
            )
            if isinstance(result, str):
                # 部分 LangChain 版本会把 dict 返回值转成 str，兜底还原
                # （本地 langchain_core 1.6.3 实测 dict 原样透传，此分支纯属防御）
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {"text": result}
            citations = None
            if name == "query_faq" and isinstance(result, dict):
                citations = build_citations(result.get("hits", [])) or None
            return ToolOutcome(name, tool_call_id, True, result,
                               make_summary(name, result), citations=citations)
        except TimeoutError:
            last_err = f"执行超时(>{context.timeout_seconds}s)"
            logger.warning("tool %s timeout (attempt %d/%d)", name, i + 1, attempts)
        except Exception as exc:  # noqa: BLE001 —— 兜底包装是设计目标
            last_err = f"{type(exc).__name__}: {exc}"
            logger.warning("tool %s failed (attempt %d/%d): %s", name, i + 1, attempts, exc)
    err_result = {"error": f"工具执行失败: {last_err}"}
    return ToolOutcome(name, tool_call_id, False, err_result, make_summary(name, err_result))
