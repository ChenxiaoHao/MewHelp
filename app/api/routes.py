import logging
from collections.abc import AsyncIterable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.core.config import get_settings
from app.context.layers import ContextStore
from app.db import crud
from app.db.engine import get_session_factory
from app.schemas.chat import (
    ChatRequest,
    ConversationEvent,
    HealthResponse,
    OrdersEvent,
    SuggestionsEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.schemas.extraction import AfterSaleExtraction, ExtractRequest
from app.schemas.refund import RefundRequest
from app.schemas.ticket import TicketCreateRequest, TicketOut
from app.schemas.knowledge import ChunkOut, FaithCaseOut, FaithCasePatch
from app.services.chat_service import get_model
from app.services.extract_service import extract_after_sale
from app.services.persistence import DBChatPersister
from app.workflows.graph import stream_graph_turn

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
            if kind == "token":
                yield ServerSentEvent(data=payload, event="token")
            elif kind == "tool_call":
                yield ServerSentEvent(
                    data=ToolCallEvent(**payload).model_dump(), event="tool_call"
                )
            elif kind == "tool_result":
                yield ServerSentEvent(
                    data=ToolResultEvent(**payload).model_dump(exclude_none=True), event="tool_result"
                )
            elif kind == "orders":  # ch06 新帧(P7):末 token 后、suggestions 前(适配层保证)
                yield ServerSentEvent(
                    data=OrdersEvent(**payload).model_dump(), event="orders"
                )
            elif kind == "suggestions":  # ch05 新帧(P4):末 token 后、done 前(适配层保证)
                yield ServerSentEvent(
                    data=SuggestionsEvent(**payload).model_dump(), event="suggestions"
                )
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
