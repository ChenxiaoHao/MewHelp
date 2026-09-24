"""ch04 原文回查与忠实度台账的 REST 契约(spec §5.5)。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class ChunkOut(BaseModel):
    id: int
    section_path: str
    category: str
    questions: str
    answer: str
    content_type: str
    prev_chunk_id: int | None
    next_chunk_id: int | None


class FaithCaseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    eval_id: str
    bucket: str
    query: str
    strategy: str
    answer: str
    reason: str
    citations: list | None
    judge_model: str | None
    status: str
    seen_count: int
    first_seen_at: datetime
    last_seen_at: datetime
    resolution: str | None
    resolved_at: datetime | None


class FaithCasePatch(BaseModel):
    status: Literal["未解决", "已解决", "无需解决"]
    resolution: str | None = None

    @model_validator(mode="after")
    def resolution_required(self):
        if self.status != "未解决" and not (self.resolution or "").strip():
            raise ValueError("标已解决/无需解决必须填写处置说明")
        return self
