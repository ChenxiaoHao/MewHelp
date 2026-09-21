"""对话挖 QA 的结构化输出 schema(spec §6 阶段一)。"""

from pydantic import BaseModel, Field


class QAItem(BaseModel):
    question: str = Field(description="用户的真实问法,完整一句,忠实于对话原文;不得编造对话里没有的问题")
    answer: str = Field(
        description="以客服回复为准的完整可执行答案,保留关键数字与条件;对话里没有可复用答案就不要输出该条"
    )


class MinedQA(BaseModel):
    items: list[QAItem] = Field(default_factory=list, description="该通会话中全部可复用 QA;没有则空列表")
