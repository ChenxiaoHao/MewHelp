import logging
from collections.abc import AsyncIterable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.core.config import get_settings
from app.db import crud
from app.db.engine import get_session_factory
from app.schemas.chat import (
    ChatRequest,
    ConversationEvent,
    HealthResponse,
    ToolCallEvent,
    ToolResultEvent,
)
from app.schemas.extraction import AfterSaleExtraction, ExtractRequest
from app.services.chat_service import get_model
from app.services.extract_service import extract_after_sale
from app.services.persistence import DBChatPersister
from app.services.tool_chat_service import stream_chat_with_tools

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

    # ---- 单轮工具编排；token/done/error 帧写法与 ch01 逐字符一致（spec §13 红线）----
    try:
        async for kind, payload in stream_chat_with_tools(
            req.messages,
            settings,
            model,
            conversation_id=conversation_id,
            persister=persister,
        ):
            if kind == "token":
                yield ServerSentEvent(data=payload, event="token")
            elif kind == "tool_call":
                yield ServerSentEvent(
                    data=ToolCallEvent(**payload).model_dump(), event="tool_call"
                )
            elif kind == "tool_result":
                yield ServerSentEvent(
                    data=ToolResultEvent(**payload).model_dump(), event="tool_result"
                )
        yield ServerSentEvent(raw_data="[DONE]", event="done")
    except Exception as exc:  # noqa: BLE001 —— SSE 惯例：错误进事件流后正常关流
        logger.exception("chat stream failed")
        yield ServerSentEvent(data={"detail": str(exc)}, event="error")


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
