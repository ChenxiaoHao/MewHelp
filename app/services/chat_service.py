from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)

from app.schemas.chat import ChatMessage


def to_langchain_messages(chat_messages: list[ChatMessage]) -> list[BaseMessage]:
    return [
        HumanMessage(content=m.content)
        if m.role == "user"
        else AIMessage(content=m.content)
        for m in chat_messages
    ]


def trim_history(history: list[BaseMessage], budget: int) -> list[BaseMessage]:
    """按 token 预算从最旧消息裁起；System 永远保留，裁剪后首条非 system 消息为 human。

    参数已按 langchain_core 1.6.3 实际签名核对（Context7 + inspect，2026-09-19）。
    """
    return trim_messages(
        history,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=budget,
        start_on="human",
        include_system=True,
    )
