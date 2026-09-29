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
import time
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
    awaiting_confirmation: bool = False  # ch08 T4:写闸拒绝→T7 react 捕获转确认流


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


# 暂时性故障白名单(spec 执行引擎节):只有这三类才重试——业务空结果/硬错
# 重试只是拖时间;OSError 覆盖网络族(含 socket 层),TimeoutError 本就子类。
TRANSIENT_ERRORS = (TimeoutError, ConnectionError, OSError)

STATUS_LABELS = {
    "COLLECTED": "已揽收", "TRANSPORT": "运输中", "DELIVERING": "派送中",
    "SIGNED": "已签收", "APPROVED": "审核通过", "RECEIVING": "收到退货中",
    "REFUNDING": "退款处理中", "CLOSED": "已关闭",
}


def format_result(spec, result):
    """MCP 结果投影(需求4「挑字段/枚举翻人话」):下划线私有键剔除、
    nodes 只留最近 5 条、状态枚举翻中文。内置工具形状一概不动(现状兼容红线)。"""
    if spec.source != "mcp" or not isinstance(result, dict):
        return result
    out = {k: v for k, v in result.items() if not str(k).startswith("_")}
    if isinstance(out.get("nodes"), list):
        out["nodes"] = out["nodes"][-5:]
    for key in ("current_status", "status"):
        if out.get(key) in STATUS_LABELS:
            out[key] = STATUS_LABELS[out[key]]
    return out


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

    # 权限闸(校验之后,闸序=校验→权限,spec 五道闸节):唯一 write=create_ticket,
    # 凭证 ticket_confirmed 在服务端 ToolContext 里,模型参数面无法伪造。
    # 拒绝不发审计——等待确认是中间态,审计按终局(confirm/cancel 路在 T7 落)。
    if spec.permission == "write" and not context.ticket_confirmed:
        err = "写操作需客户在预览卡片确认后才执行"
        return ToolOutcome(name, tool_call_id, False, {"error": err},
                           "等待客户确认", awaiting_confirmation=True)

    t0 = time.monotonic()
    config = {"configurable": {"conversation_id": context.conversation_id}}
    # 需求4:写默认不自动重试(重复执行比失败更糟);读按白名单重试
    attempts = 1 if spec.permission == "write" else max(1, context.max_retries + 1)
    status, retries_used, last_err, result = "失败", 0, "未知错误", None
    for i in range(attempts):
        retries_used = i
        try:
            result = await asyncio.wait_for(
                spec.tool.ainvoke(args, config=config), timeout=context.timeout_seconds
            )
            status = "成功"
            break
        except TimeoutError:
            status, last_err = "超时", f"执行超时(>{context.timeout_seconds}s)"
            logger.warning("tool %s timeout (attempt %d/%d)", name, i + 1, attempts)
        except TRANSIENT_ERRORS as exc:
            status, last_err = "失败", f"{type(exc).__name__}: {exc}"
            logger.warning("tool %s transient fail (attempt %d/%d): %s",
                           name, i + 1, attempts, exc)
        except Exception as exc:  # noqa: BLE001 —— 非暂时性:业务硬错,重试无意义
            status, last_err = "失败", f"{type(exc).__name__}: {exc}"
            break
    duration_ms = int((time.monotonic() - t0) * 1000)
    if status == "成功":
        if spec.source == "mcp" and isinstance(result, list) and len(result) == 1:
            # T6 真链路实证:adapters 0.3.2 ainvoke 回 LangChain content 块列表
            # (结构化 dict 在 artifact 里),唯一 text 块为合法 JSON 时还原成 dict,
            # 格式化面才能挑字段/翻人话;非 JSON/多块原样透传(需求4 空结果非异常)。
            blk = result[0]
            if (isinstance(blk, dict) and blk.get("type") == "text"
                    and isinstance(blk.get("text"), str)):
                try:
                    result = json.loads(blk["text"])
                except json.JSONDecodeError:
                    pass
        if isinstance(result, str):
            # 部分 LangChain 版本会把 dict 返回值转成 str，兜底还原
            # （本地 langchain_core 1.6.3 实测 dict 原样透传，此分支纯属防御;
            #  MCP 工具返 str 同样走这里——Review Focus 3）
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {"text": result}
        result = format_result(spec, result)
        citations = (build_citations(result.get("hits", [])) or None
                     if name == "query_faq" and isinstance(result, dict) else None)
        summary = make_summary(name, result)
        await _emit(context, _record(context, spec, tool_call_id, args, "成功",
                                     summary, retries=retries_used,
                                     duration_ms=duration_ms))
        return ToolOutcome(name, tool_call_id, True, result, summary,
                           citations=citations)
    if status == "超时" and spec.permission == "write":
        last_err += "(写操作超时未自动重试,请人工核实是否已执行)"
    err_result = {"error": f"工具执行失败: {last_err}"}
    await _emit(context, _record(context, spec, tool_call_id, args, status,
                                 make_summary(name, err_result), last_err,
                                 retries=retries_used, duration_ms=duration_ms))
    return ToolOutcome(name, tool_call_id, False, err_result,
                       make_summary(name, err_result))
