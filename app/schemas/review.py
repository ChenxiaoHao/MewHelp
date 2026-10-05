"""ch09 T9:审核队列三路由的请求/响应件。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ReviewQueueItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    normalized_question: str
    ai_suggested_answer: str | None
    occurrence_count: int
    review_status: str
    created_at: datetime
    updated_at: datetime


class ReviewSource(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    raw_question: str
    source: str
    reason: str | None
    retrieved_chunks: list | None
    created_at: datetime


class ReviewDetail(ReviewQueueItem):
    sources: list[ReviewSource] = Field(default_factory=list)


class ReviewPatch(BaseModel):
    """流转目标只认 通过|驳回;通过必带 approved_answer(端点校验 422)。"""

    status: Literal["通过", "驳回"]
    approved_answer: str | None = None


class ReviewOut(BaseModel):
    id: int
    review_status: str
    chunk_id: int | None = None
