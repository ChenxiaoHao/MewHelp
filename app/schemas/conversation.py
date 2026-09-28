"""只读会话 API 响应面(ch07 T9,spec「HTTP API(P8 口径)」)。

preview=首条 user 截 40 字(crud 面截好,API 不二次加工);
summarized=summary_upto_msg_id IS NOT NULL(层2 已有投影=摘要过);
MessageItem.tool_calls=spec「tool_calls 摘要面」透传(brief 未列,冲突按 spec 解,
前端 tool 行浅标识渲染用)。两接口只读,无任何写路径。
"""

from datetime import datetime

from pydantic import BaseModel


class ConversationItem(BaseModel):
    id: int
    created_at: datetime
    preview: str | None
    summarized: bool


class ConversationList(BaseModel):
    items: list[ConversationItem]


class MessageItem(BaseModel):
    id: int
    role: str
    content: str | None
    tool_calls: list | None = None
    created_at: datetime
