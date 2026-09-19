from enum import Enum

from pydantic import BaseModel, Field


class IssueType(str, Enum):
    refund = "退款"
    return_goods = "退货"  # return 是 Python 关键字，用 return_goods
    exchange = "换货"
    logistics = "物流"
    quality = "质量问题"
    other = "其他"


class AfterSaleExtraction(BaseModel):
    """从用户售后描述中提取的结构化字段。"""

    order_id: str | None = Field(None, description="订单号，用户未提供时为 None")
    issue_type: IssueType = Field(description="诉求类型")
    expected_solution: str = Field(description="用户期望的处理方案")


class ExtractRequest(BaseModel):
    description: str = Field(min_length=1, description="用户的售后描述原文")
