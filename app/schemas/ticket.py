from pydantic import BaseModel, Field


class TicketCreateRequest(BaseModel):
    """POST /api/tickets 请求体（ch05 需求 8：前端「建工单」按钮专用通道）。"""

    conversation_id: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1)


class TicketOut(BaseModel):
    ticket_no: str
