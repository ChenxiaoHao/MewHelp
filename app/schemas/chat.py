from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.context.budget import estimate_text
from app.core.config import get_settings


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
        # ch07 需求4-P5:当前句超 max_user_input_tokens → 校验失败(路由 422,不截不 500)
        limit = get_settings().max_user_input_tokens
        used = estimate_text(self.messages[-1].content)
        if used > limit:
            raise ValueError(f"当前输入过长(≈{used} token > 上限 {limit})")
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
    """ch05 操作建议项：转人工/建工单两按钮各自独立（需求 8），前端自选渲染。
    ch06 T5 扩 refund_apply（退款申请表单入口，拍板 P7）。"""

    action: Literal["transfer_human", "create_ticket", "refund_apply"]
    label: str


class SuggestionsEvent(BaseModel):
    """SSE `suggestions` 帧 data：后端只发建议不代决策（spec「SSE 契约」节）。"""

    items: list[Suggestion]


class OrderCard(BaseModel):
    """ch06 订单选择器卡片（临时 UI 数据,不落消息历史;与 query_order 同播种）。"""

    order_id: str
    status: str
    amount: float
    created_at: str
    items: list[str]          # 「商品名 ×数量」显示串


class OrdersEvent(BaseModel):
    """SSE `orders` 帧 data：退款流程缺单号时弹可点卡片（需求 6/拍板 P5）。"""

    items: list[OrderCard]


class TicketPreviewEvent(BaseModel):
    """SSE `ticket_preview` 帧 data：ch08 确认流预览卡（图 interrupt 挂起，等客户在
    卡片上决策；confirm 端点回传后续播，需求7）。"""

    tool_call_id: str
    ticket_type: str
    description: str
    conversation_id: int | None = None
