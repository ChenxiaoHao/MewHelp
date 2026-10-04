"""ch09 观测基座:Langfuse handler 工厂 + 编译处单挂(spec「观测链」节)。

三键(langfuse_host/public_key/secret_key)任一缺 → enabled False →
观测整体短路,图返回形状与 ch08 终态一致(Review Focus 1:零行为漂移)。

SDK v4 的客户端单例首建时只读进程环境变量:工厂先把 Settings(.env 优先)
里的三键写入 os.environ,再构造 CallbackHandler——保证 .env 形态对 shell
env 未导出的场景同样生效。
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

_KEYS = ("langfuse_host", "langfuse_public_key", "langfuse_secret_key")
_ENV = {"langfuse_host": "LANGFUSE_HOST",
        "langfuse_public_key": "LANGFUSE_PUBLIC_KEY",
        "langfuse_secret_key": "LANGFUSE_SECRET_KEY"}


def enabled(settings) -> bool:
    return all(getattr(settings, k, "") for k in _KEYS)


def _bypass_proxy(host: str) -> None:
    """Windows 注册表系统代理会被 httpx(trust_env)套到 localhost 请求上→502。

    把观测服务主机名并入 NO_PROXY(进程内),自部署场景(本机端口/内网名)
    直连,不受用户机代理软件规则影响。
    """
    try:
        from urllib.parse import urlparse
        netloc = urlparse(host).hostname or ""
    except ValueError:
        return
    if not netloc:
        return
    for var in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(var, "")
        items = {s.strip() for s in cur.split(",") if s.strip()}
        if netloc not in items:
            os.environ[var] = f"{cur},{netloc}" if cur else netloc


def build_handler(settings, trace_id=None):
    """有键=构造一次挂图编译处;无键=None(调用方不挂)。

    SDK v4 事实:CallbackHandler 按 public_key 找已注册客户端,未先建
    Langfuse() 则静默丢 span——工厂必须先实例化客户端再挂 handler。
    trace_id 非空=整轮 span 落进预生成 trace(归因写回的定向通道,T2)。
    """
    if not enabled(settings):
        return None
    for k in _KEYS:
        os.environ[_ENV[k]] = getattr(settings, k)
    _bypass_proxy(settings.langfuse_host)
    from langfuse import Langfuse  # 惰性:无键环境不触 SDK
    from langfuse.langchain import CallbackHandler
    Langfuse(public_key=settings.langfuse_public_key,
             secret_key=settings.langfuse_secret_key,
             host=settings.langfuse_host)
    if trace_id:
        return CallbackHandler(public_key=settings.langfuse_public_key,
                               trace_context={"trace_id": trace_id})
    return CallbackHandler(public_key=settings.langfuse_public_key)


def turn_trace_id(settings, seed=None):
    """逐轮预生成 trace_id;缺键=None(观测短路,与零漂移语义一致)。"""
    if not enabled(settings):
        return None
    from langfuse import Langfuse
    return Langfuse.create_trace_id(seed=seed)


async def attach_trace_intent(trace_id, settings, intent, confidence):
    """流末把 intent 写成 trace tag + metadata;任一前提不满足=静默 no-op。

    定一道(spec 备道转正):v4 无 update_current_trace,走 trace-create
    ingestion 事件按 id upsert 合并——与 SDK 内部 batch_evaluation 同法。
    """
    if not (enabled(settings) and trace_id and intent):
        return
    try:
        await asyncio.to_thread(_update_trace_tags, trace_id, intent, confidence)
    except Exception:  # noqa: BLE001 —— 归因失败不毁整轮(观测面全吞)
        logger.warning("trace intent attach failed tid=%s intent=%s",
                       trace_id, intent, exc_info=True)


def _update_trace_tags(trace_id, intent, confidence):
    """同步低层更新:一条 trace-create 事件按 id 合并 tags+metadata。"""
    from datetime import datetime, timezone
    from langfuse import get_client
    from langfuse.api.ingestion.types import IngestionEvent_TraceCreate, TraceBody
    client = get_client(public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"))
    body = TraceBody(id=trace_id, tags=[f"intent:{intent}"],
                     metadata={"intent": intent,
                               "intent_confidence": confidence})
    event = IngestionEvent_TraceCreate(
        id=client.create_trace_id(), body=body,
        timestamp=datetime.now(timezone.utc).isoformat(), type="trace-create")
    client.api.ingestion.batch(batch=[event])


def augment_turn_cfg(cfg: dict, settings) -> dict:
    """enabled 时给每轮 config 补 trace 归因键(session/run_name),原地改并回传。

    langfuse_session_id 直接用 thread_id(conv-{id}/anon-{uuid} 两形态同源)。
    """
    if not enabled(settings):
        return cfg
    cfg["run_name"] = "chat_turn"
    meta = cfg.setdefault("metadata", {})
    meta["langfuse_session_id"] = cfg["configurable"].get("thread_id")
    return cfg
