from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.prompts.customer_service import CUSTOMER_SERVICE_PROMPT
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

    保底规则：若预算小到连最新一条消息都装不下（trim_messages 会把它裁掉、
    只剩 system），则强制返回 [system...] + [最后一条]——绝不丢用户当前问题。
    """
    trimmed = trim_messages(
        history,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=budget,
        start_on="human",
        include_system=True,
    )
    if history and all(t is not history[-1] for t in trimmed):
        system_msgs = [m for m in history if isinstance(m, SystemMessage)]
        return system_msgs + [history[-1]]
    return trimmed


def get_model(settings: Settings) -> ChatOpenAI:
    """OpenAI 协议直连上游；换 GPT/Claude/DeepSeek/Ollama 只改 .env。"""
    return ChatOpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.model_name,
        temperature=settings.temperature,
    )


def build_messages(
    chat_messages: list[ChatMessage], settings: Settings
) -> list[BaseMessage]:
    """客户端历史 → LangChain 消息 → 拼 System Prompt → 按预算裁剪。"""
    history = to_langchain_messages(chat_messages)
    rendered = CUSTOMER_SERVICE_PROMPT.invoke({"messages": history}).to_messages()
    return trim_history(rendered, settings.history_token_budget)


async def stream_chat(messages: list[BaseMessage], model) -> AsyncIterator[str]:
    """流式对话：yield 增量文本。chunk.text 为 langchain 1.x 当前 API（Context7 已核对）。"""
    async for chunk in model.astream(messages):
        if chunk.text:
            yield chunk.text
