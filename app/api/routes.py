import logging
from collections.abc import AsyncIterable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.core.config import get_settings
from app.schemas.chat import ChatRequest, HealthResponse
from app.schemas.extraction import AfterSaleExtraction, ExtractRequest
from app.services.chat_service import build_messages, get_model, stream_chat
from app.services.extract_service import extract_after_sale

logger = logging.getLogger(__name__)
router = APIRouter()


def dep_chat_model():
    """模型依赖注入点：测试用 app.dependency_overrides 替换为 Fake。"""
    return get_model(get_settings())


@router.post("/api/chat/stream", response_class=EventSourceResponse)
async def chat_stream(
    req: ChatRequest, model=Depends(dep_chat_model)
) -> AsyncIterable[ServerSentEvent]:
    try:
        settings = get_settings()
        messages = build_messages(req.messages, settings)
        async for text in stream_chat(messages, model):
            yield ServerSentEvent(data=text, event="token")
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
async def health():
    s = get_settings()
    return HealthResponse(
        status="ok", model=s.model_name, history_token_budget=s.history_token_budget
    )
