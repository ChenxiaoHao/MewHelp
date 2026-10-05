import asyncio
import logging
import time
from collections.abc import AsyncIterable

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.core.config import get_settings
from app.context.layers import ContextStore
from app.db import crud
from app.db.engine import get_session_factory
from app.schemas.chat import (
    ChatRequest,
    ConversationEvent,
    FeedbackRequest,
    HealthResponse,
    OrdersEvent,
    SuggestionsEvent,
    TicketPreviewEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.schemas.conversation import ConversationItem, ConversationList, MessageItem
from app.schemas.extraction import AfterSaleExtraction, ExtractRequest
from app.schemas.refund import RefundRequest
from app.schemas.ticket import TicketConfirmRequest, TicketCreateRequest, TicketOut
from app.schemas.knowledge import ChunkOut, FaithCaseOut, FaithCasePatch
from app.schemas.review import (
    ReviewDetail,
    ReviewOut,
    ReviewPatch,
    ReviewQueueItem,
    ReviewSource,
)
from app.services import flywheel, review
from app.services.chat_service import get_model
from app.services.extract_service import extract_after_sale
from app.services.persistence import DBChatPersister
from app.workflows.graph import build_graph, stream_graph_turn, thread_lock

logger = logging.getLogger(__name__)
router = APIRouter()


def dep_settings():
    """配置依赖注入点：测试用 app.dependency_overrides 替换。"""
    return get_settings()


def dep_chat_model(settings=Depends(dep_settings)):
    """模型依赖注入点：测试用 app.dependency_overrides 替换为 Fake。"""
    return get_model(settings)


async def dep_db_session():
    """DB 会话依赖注入点：测试用 dependency_overrides 替换。
    引擎未初始化（ch01 单测环境/未配 MySQL）→ yield None，路由降级：
    不推 conversation 帧、不落库，token/done/error 行为与 ch01 完全一致。"""
    try:
        factory = get_session_factory()
    except RuntimeError:
        yield None
        return
    async with factory() as session:
        yield session


def _sse_event(kind: str, payload) -> ServerSentEvent | None:
    """帧→SSE 映射 helper（ch08 T8 抽取自 chat_stream elif 链，chat/confirm 两端复用）。

    token/done/error 帧字节面与 ch01 逐字符一致（spec 红线）；未知 kind 静默丢
    （与旧 elif 链行为一致）。"""
    if kind == "token":
        return ServerSentEvent(data=payload, event="token")
    if kind == "tool_call":
        return ServerSentEvent(data=ToolCallEvent(**payload).model_dump(), event="tool_call")
    if kind == "tool_result":
        return ServerSentEvent(
            data=ToolResultEvent(**payload).model_dump(exclude_none=True), event="tool_result"
        )
    if kind == "orders":  # ch06 新帧(P7):末 token 后、suggestions 前(适配层保证)
        return ServerSentEvent(data=OrdersEvent(**payload).model_dump(), event="orders")
    if kind == "suggestions":  # ch05 新帧(P4):末 token 后、done 前(适配层保证)
        return ServerSentEvent(data=SuggestionsEvent(**payload).model_dump(), event="suggestions")
    if kind == "ticket_preview":  # ch08 需求7:预览卡帧,其后无 done 语义由编排层保证
        return ServerSentEvent(data=TicketPreviewEvent(**payload).model_dump(), event="ticket_preview")
    return None


# M2-I3:「409 预检 + resume」两段 await 之间无原子性——卡片双击并发两发都在对方
# 写 checkpoint 前通过预检,同点重放 ticket_confirm → 唯一 write 双落(需求4 红线)。
# 修法:短临界区(检+加之间无 await)单入口;端点生成器 finally 释放。
# dep 通过后生成器未及启动的断连泄漏由 TTL 自愈(120s≫resume 实际百毫秒级;
# 过期只放松「在飞拦截」不放松「pending 检查」——重复建单闸不因此失效)。
_CONFIRM_INFLIGHT_TTL = 120.0
_confirm_inflight: dict[int, float] = {}
_confirm_guard = asyncio.Lock()


async def dep_confirm_pending(
    req: TicketConfirmRequest,
    settings=Depends(dep_settings),
    model=Depends(dep_chat_model),
    session=Depends(dep_db_session),
):
    """503/409 预检放依赖层:端点流式函数体在响应头发送后才执行,体内 raise
    改不了 HTTP 状态码;依赖在发送前 await,409/503 走标准 HTTP 错误面(需求7)。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    graph = build_graph(settings, model)
    cfg = {"configurable": {"thread_id": f"conv-{req.conversation_id}",
                            "conversation_id": req.conversation_id,
                            "persister": None, "ctx_store": None}}
    # 终审 I-1:「查 pending+标 inflight」与新消息轮的「查 pending+drain」共持
    # per-thread 锁(锁只跨检查段,resume 头段锁在 stream_graph_turn 内接力)。
    async with thread_lock(cfg["configurable"]["thread_id"]):
        snap = await graph.aget_state(cfg)
        if not (getattr(snap, "interrupts", ()) or ()):
            raise HTTPException(status_code=409, detail="确认已过期,请重新发起建单")
        now = time.monotonic()
        async with _confirm_guard:
            started = _confirm_inflight.get(req.conversation_id)
            if started is not None and now - started < _CONFIRM_INFLIGHT_TTL:
                raise HTTPException(status_code=409, detail="已有确认在处理中，请稍候")
            _confirm_inflight[req.conversation_id] = now
    return session


@router.post("/api/tickets/confirm", response_class=EventSourceResponse)
async def ticket_confirm(
    req: TicketConfirmRequest,
    settings=Depends(dep_settings),
    model=Depends(dep_chat_model),
    session=Depends(dep_confirm_pending),
) -> AsyncIterable[ServerSentEvent]:
    """ch08 需求7:预览卡片回传 → Command(resume) 同线程续跑,SSE 续播答复。

    409=线程无挂起 interrupt(卡片过期/已被消费,预检挡);
    422=decision 乱值(请求体校验挡下,interrupt 不被消耗——Review Focus 2)。"""
    persister = DBChatPersister(session, req.conversation_id)
    try:
        async for kind, payload in stream_graph_turn(
                None, settings, model, conversation_id=req.conversation_id,
                persister=persister, resume_value=req.decision):
            ev = _sse_event(kind, payload)
            if ev is not None:
                yield ev
        yield ServerSentEvent(raw_data="[DONE]", event="done")
    except Exception as exc:  # noqa: BLE001 —— SSE 惯例：错误进事件流后正常关流
        logger.exception("confirm resume failed")
        yield ServerSentEvent(data={"detail": str(exc)}, event="error")
    finally:
        _confirm_inflight.pop(req.conversation_id, None)   # M2-I3 释放单入口


@router.post("/api/chat/stream", response_class=EventSourceResponse)
async def chat_stream(
    req: ChatRequest,
    settings=Depends(dep_settings),
    model=Depends(dep_chat_model),
    session=Depends(dep_db_session),
) -> AsyncIterable[ServerSentEvent]:
    # ---- on_turn_start 挂点（spec §7）：建/复用会话 + user 行 + conversation 帧 ----
    conversation_id = req.conversation_id
    persister = None
    if session is not None:
        try:
            conv = await crud.create_or_get_conversation(
                session, conversation_id, settings.demo_user_id
            )
            conversation_id = conv.id
            await crud.add_message(
                session, conversation_id, "user", content=req.messages[-1].content
            )
            persister = DBChatPersister(session, conversation_id)
            yield ServerSentEvent(
                data=ConversationEvent(conversation_id=conversation_id).model_dump(),
                event="conversation",
            )
        except Exception:  # noqa: BLE001 —— 运行期落库失败只降级，不挡聊天（spec §9）
            logger.warning("conversation bootstrap failed; continue without persistence", exc_info=True)
            persister = None

    # ---- ch05 Graph 编排；token/done/error 帧写法与 ch01 逐字符一致（spec 红线）。
    # ch04 的 stream_chat_with_tools 不再接线但保留（拍板 P5=ch04 回归基线）。
    # ch07:引擎在线且有会话 → ContextStore 入 configurable(ctx/coref/agent 三消费面;
    # 引擎降级面 store=None=全 passthrough,与 ch01 行为逐字对齐)。----
    ctx_store = None
    if session is not None and conversation_id is not None:
        try:
            ctx_store = ContextStore(get_session_factory(), conversation_id, settings)
        except RuntimeError:
            # 假 session 覆盖面(ch05/06 流测)引擎未初始化:取厂即炸 → store 降级,
            # 本轮走 ch01 旧路径——引擎降级面与 store=None 语义一致(Ruling)。
            logger.warning("ctx store unavailable; context layers degraded", exc_info=True)
    try:
        async for kind, payload in stream_graph_turn(
            req.messages,
            settings,
            model,
            conversation_id=conversation_id,
            persister=persister,
            ctx_store=ctx_store,
        ):
            ev = _sse_event(kind, payload)
            if ev is not None:
                yield ev
        yield ServerSentEvent(raw_data="[DONE]", event="done")
    except Exception as exc:  # noqa: BLE001 —— SSE 惯例：错误进事件流后正常关流
        logger.exception("chat stream failed")
        yield ServerSentEvent(data={"detail": str(exc)}, event="error")


@router.post("/api/tickets", response_model=TicketOut, status_code=201)
async def create_ticket(
    req: TicketCreateRequest, session=Depends(dep_db_session)
) -> TicketOut:
    """ch05 需求 8：前端「建工单」按钮点下才写 tickets 表；与「转人工」互不绑定
    （crud 层 status 副作用已移除，D4）。ticket_type 固定「咨询」——请求面不带类型。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    row = await crud.create_ticket(
        session,
        conversation_id=req.conversation_id,
        description=f"【{req.title}】{req.content}",
        ticket_type="咨询",
    )
    return TicketOut(ticket_no=row.ticket_no)


@router.post("/api/refunds", response_model=TicketOut, status_code=201)
async def create_refund(req: RefundRequest, session=Depends(dep_db_session)) -> TicketOut:
    """ch06 需求 6/拍板 P6：退款表单提交 → tickets 表，ticket_type=「售后」。
    描述服务端拼装；reason 只收固定四类（前端下拉），自由文案进不来。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    row = await crud.create_ticket(
        session,
        conversation_id=req.conversation_id,
        description=f"【退款申请】订单 {req.order_id}｜原因：{req.reason}",
        ticket_type="售后",
    )
    return TicketOut(ticket_no=row.ticket_no)


@router.post("/api/extract", response_model=AfterSaleExtraction)
async def extract(req: ExtractRequest, model=Depends(dep_chat_model)):
    try:
        return await extract_after_sale(req.description, model)
    except Exception as exc:  # noqa: BLE001
        logger.exception("extract failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/api/health", response_model=HealthResponse)
async def health(settings=Depends(dep_settings)):
    return HealthResponse(
        status="ok",
        model=settings.model_name,
        history_token_budget=settings.history_token_budget,
    )


@router.post("/api/feedback")
async def cast_feedback(req: FeedbackRequest, session=Depends(dep_db_session)):
    """ch09 T6 👎 后端化:反查第 seq(1 基)个可见 assistant 轮→带快照落池
    (source=user_feedback,重复👎允许);👍=204 记账面留给评估链。
    同步面不吞错——落池失败必须可见(裁决见 ledger;M2-I1/I2 契约钉)。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    if req.vote == "up":
        return Response(status_code=204)
    anchor, raw_question = await crud.find_feedback_anchor(
        session, req.conversation_id, req.seq)
    if anchor is None:
        raise HTTPException(status_code=404, detail="找不到该轮(会话不存在或 seq 越界)")
    row_id = await crud.add_low_confidence_question(
        session, conversation_id=req.conversation_id, raw_question=raw_question,
        source="user_feedback", reason=f"seq={req.seq}",
        retrieved_chunks=anchor.retrieval_snapshot)
    flywheel.spawn_process(row_id)  # T7 拍板 2A:👎 路落池成功即触发(不走吞错漏斗)
    return {"pooled": True}


@router.get("/api/review_queue", response_model=list[ReviewQueueItem])
async def list_review_queue(status: str | None = None,
                            session=Depends(dep_db_session)):
    """审核队列列表(T9):occurrence desc,可按状态过滤。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    return await crud.list_review_queue(session, status=status)


@router.get("/api/review_queue/{rq_id}/detail", response_model=ReviewDetail)
async def get_review_detail(rq_id: int, session=Depends(dep_db_session)):
    """详情(T10 弹层数据面):行 + 归并 LCQ 原话/当轮召回快照。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    row = await crud.get_review_row(session, rq_id)
    if row is None:
        raise HTTPException(status_code=404, detail="队列行不存在")
    detail = ReviewDetail.model_validate(row)
    detail.sources = [ReviewSource.model_validate(s)
                      for s in await crud.review_sources(session, rq_id)]
    return detail


@router.patch("/api/review_queue/{rq_id}", response_model=ReviewOut)
async def patch_review_queue(rq_id: int, req: ReviewPatch,
                             session=Depends(dep_db_session)):
    """状态机(T9):仅 待审→通过|驳回;通过=同请求内核准回写双落成功才置态,
    KB 写失败 502 状态留待审(Focus 5——半成功不许存在)。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    row = await crud.get_review_row(session, rq_id)
    if row is None:
        raise HTTPException(status_code=404, detail="队列行不存在")
    if row.review_status != "待审":
        raise HTTPException(status_code=422,
                            detail=f"非法流转:{row.review_status}→{req.status}")
    if req.status == "通过":
        answer = (req.approved_answer or "").strip()
        if not answer:
            raise HTTPException(status_code=422, detail="通过必须带核准答案")
        try:
            chunk_id = await review.publish_approved(session, row, answer)
        except Exception as exc:  # noqa: BLE001 —— 半成功闸:写不成就不置态,502 可见
            logger.warning("review publish failed id=%s", rq_id, exc_info=True)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return ReviewOut(id=row.id, review_status="通过", chunk_id=chunk_id)
    row = await crud.reject_review(session, rq_id)
    return ReviewOut(id=row.id, review_status="驳回")


@router.get("/api/chunks/{chunk_id}", response_model=ChunkOut)
async def get_chunk(chunk_id: int, session=Depends(dep_db_session)):
    """引用弹窗原文(T13)。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    row = await crud.get_chunk(session, chunk_id)
    if row is None:
        raise HTTPException(status_code=404, detail="chunk 不存在")
    return row


@router.get("/api/faith_cases", response_model=list[FaithCaseOut])
async def list_faith_cases(status: str | None = None, bucket: str | None = None,
                           session=Depends(dep_db_session)):
    """台账页列表(T14):last_seen_at 降序,可按状态/桶过滤。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    return await crud.list_faith_cases(session, status=status, bucket=bucket)


@router.patch("/api/faith_cases/{case_id}", response_model=FaithCaseOut)
async def patch_faith_case(case_id: int, req: FaithCasePatch, session=Depends(dep_db_session)):
    """处置流转:非「未解决」必须带 resolution(schema 422);「未解决」强制清处置说明。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    row = await crud.set_faith_case_status(session, case_id, req.status, req.resolution)
    if row is None:
        raise HTTPException(status_code=404, detail="个案不存在")
    return row


@router.get("/api/conversations", response_model=ConversationList)
async def list_conversations(settings=Depends(dep_settings),
                             session=Depends(dep_db_session)):
    """侧栏列表(ch07 T9,只读):id 降序+首问预览截 40 字+已摘要标记,demo_user_id 面(P8)。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    rows = await crud.list_user_conversations(session, settings.demo_user_id)
    return ConversationList(items=[
        ConversationItem(id=r[0], created_at=r[1], preview=r[2], summarized=r[3])
        for r in rows
    ])


@router.get("/api/conversations/{conversation_id}/messages",
            response_model=list[MessageItem])
async def get_conversation_messages(conversation_id: int,
                                    settings=Depends(dep_settings),
                                    session=Depends(dep_db_session)):
    """消息回载(ch07 T9,只读):全行升序含 tool 行(P8 口径);不存在与非属主同答
    404(不泄归属面);limit=10000 显式——I2 同律,静默截断=前端少历史还装全。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    if await crud.get_conversation_for_user(
            session, conversation_id, settings.demo_user_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    rows = await crud.list_messages_after(session, conversation_id, 0, limit=10000)
    return [MessageItem(id=m.id, role=m.role, content=m.content,
                        tool_calls=m.tool_calls, created_at=m.created_at)
            for m in rows]
