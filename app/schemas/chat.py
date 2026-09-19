from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)

    @model_validator(mode="after")
    def last_message_must_be_user(self) -> "ChatRequest":
        if self.messages[-1].role != "user":
            raise ValueError("messages 最后一条的 role 必须是 user")
        return self


class HealthResponse(BaseModel):
    status: str
    model: str
    history_token_budget: int
