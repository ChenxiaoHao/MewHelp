from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    conversation_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def last_message_must_be_user(self) -> "ChatRequest":
        if self.messages[-1].role != "user":
            raise ValueError("messages 最后一条的 role 必须是 user")
        return self


class HealthResponse(BaseModel):
    status: str
    model: str
    history_token_budget: int


class ConversationEvent(BaseModel):
    """SSE `conversation` 帧 data：服务端告知本轮会话 id（每轮都推，幂等）。"""

    conversation_id: int


class ToolCallEvent(BaseModel):
    """SSE `tool_call` 帧 data：模型决定调用工具（前端渲染「调用中」徽章）。"""

    id: str
    name: str
    args: dict


class ToolResultEvent(BaseModel):
    """SSE `tool_result` 帧 data：工具执行结束（前端更新徽章 ✓/✗ + summary 提示）。"""

    id: str
    name: str
    ok: bool
    summary: str
    citations: list[dict] | None = None  # ch04 可选增列;routes 帧 dump 用 exclude_none,无引用时键不出现


class Suggestion(BaseModel):
    """ch05 操作建议项：转人工/建工单两按钮各自独立（需求 8），前端自选渲染。"""

    action: Literal["transfer_human", "create_ticket"]
    label: str


class SuggestionsEvent(BaseModel):
    """SSE `suggestions` 帧 data：后端只发建议不代决策（spec「SSE 契约」节）。"""

    items: list[Suggestion]
