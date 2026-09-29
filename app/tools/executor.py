"""统一执行引擎(spec 五道闸节)单点漏斗:ToolSpec 入,ToolOutcome 出。

T3 起签名收 ToolSpec(查找职责移到调用方),校验闸在最先:
jsonschema Draft202012Validator 拦下不抛异常——错误文本包成 ok=False
结果回灌模型,让它礼貌追问,而不是炸 SSE 流。
失败兜底原则(ch02 起沿用)：execute_tool 绝不向上抛。
审计:校验拦下/权限拒绝即时落(T3),成功/失败/超时全漏斗在 T5 接线。
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from app.rag.hit_format import build_citations
from app.tools.audit import AuditRecord, db_audit_sink

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    conversation_id: int | None = None
    timeout_seconds: float = 5.0
    max_retries: int = 1
    audit_sink: Any = None            # None=db_audit_sink(T1 默认落库面)
    ticket_confirmed: bool = False    # T4 权限闸:T7 确认流恢复轮凭证


@dataclass
class ToolOutcome:
    name: str
    tool_call_id: str
    ok: bool
    result: Any  # dict：成功=工具返回；失败={"error": "..."}
    summary: str  # ≤80 字符：前端徽章文案 / tool 消息落库 content
    citations: list | None = None  # ch04: query_faq 命中集引用(前端弹窗数据);其余工具恒 None


_SCHEMA_CACHE: dict[int, dict] = {}


def tool_json_schema(tool) -> dict:
    """模型可见参数面 schema;tool_call_schema 是 langchain 剔除注入参数
    (config/RunnableConfig)后的调用形状——校验必须用它,不然 config 会被当必填。"""
    sc = getattr(tool, "tool_call_schema", None)
    if isinstance(sc, type):
        try:
            return sc.model_json_schema()
        except AttributeError:
            pass
    return {"type": "object", "properties": tool.args or {}}   # 兜底拼装


def validate_args(spec, args: dict) -> str | None:
    """校验闸:返回 None=放行;否则中文错误文本(≤512,直接回灌+落审计)。"""
    key = id(spec.tool)
    schema = _SCHEMA_CACHE.setdefault(key, tool_json_schema(spec.tool))
    errs = sorted(Draft202012Validator(schema).iter_errors(args),
                  key=lambda e: list(e.path))
    if not errs:
        return None
    e = errs[0]
    loc = "/".join(str(p) for p in e.path) or "(root)"
    return f"参数不合法: {loc} {e.message}"[:512]


async def _emit(ctx: ToolContext, record: AuditRecord) -> None:
    sink = ctx.audit_sink if ctx.audit_sink is not None else db_audit_sink
    try:
        await sink(record)
    except Exception:  # noqa: BLE001 —— 审计永不反拦执行(需求5)
        logger.warning("audit emit failed name=%s", record.tool_name, exc_info=True)


def _record(ctx, spec, tool_call_id, args, status, summary, error=None,
            retries=0, duration_ms=0) -> AuditRecord:
    return AuditRecord(conversation_id=ctx.conversation_id, tool_call_id=tool_call_id,
                       tool_name=spec.name, tool_source=spec.source,
                       mcp_server=spec.mcp_server, arguments=args,
                       result_summary=summary[:500], status=status,
                       error_message=error[:512] if error else None,
                       retry_count=retries, duration_ms=duration_ms)


async def audit_denied(ctx: ToolContext, name: str, tool_call_id: str,
                       args: dict, reason: str = "工具未登记，拒绝执行") -> None:
    """幻觉未登记调用/权限闸拒绝共用审计面(Review Focus 6):
    tool_source 恒 builtin——查无此件时来源无从谈起,按我方命名空间拒绝记。"""
    rec = AuditRecord(conversation_id=ctx.conversation_id, tool_call_id=tool_call_id,
                      tool_name=name, tool_source="builtin", mcp_server=None,
                      arguments=args, result_summary=reason[:500], status="权限拒绝",
                      error_message=reason[:512], retry_count=0, duration_ms=0)
    await _emit(ctx, rec)


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
    spec, args: dict, tool_call_id: str, context: ToolContext
) -> ToolOutcome:
    name = spec.name
    err = validate_args(spec, args)
    if err:
        await _emit(context, _record(context, spec, tool_call_id, args,
                                     "校验拦下", err, err))
        return ToolOutcome(name, tool_call_id, False, {"error": err},
                           make_summary(name, {"error": err}))

    tool = spec.tool
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
