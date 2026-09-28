"""退款申请请求面(ch06 需求 6/拍板 P6)。

reason 固定四类=表单下拉,服务端拒绝自由文案;描述由 routes 拼装,
请求面不带 title/content(与 tickets 通道的差异点)。
"""

from typing import Literal

from pydantic import BaseModel, Field

RefundReason = Literal["七天无理由", "商品质量问题", "拍错多拍", "其他"]


class RefundRequest(BaseModel):
    conversation_id: int = Field(ge=1)
    order_id: str = Field(pattern=r"^\d{3,}$")
    reason: RefundReason
