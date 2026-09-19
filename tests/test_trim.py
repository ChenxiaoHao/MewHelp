from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _conv(n_turns: int, filler: str = "这是一段用于填充token的中文文本内容。"):
    msgs = [SystemMessage(content="system prompt")]
    for i in range(n_turns):
        msgs.append(HumanMessage(content=f"{filler}第{i}轮问题"))
        msgs.append(AIMessage(content=f"{filler}第{i}轮回答"))
    return msgs


def test_trim_keeps_system_and_latest():
    from app.services.chat_service import trim_history

    msgs = _conv(10)
    trimmed = trim_history(msgs, budget=60)
    assert isinstance(trimmed[0], SystemMessage)
    assert trimmed[-1].content == msgs[-1].content
    assert len(trimmed) < len(msgs)


def test_trim_first_non_system_is_human():
    from app.services.chat_service import trim_history

    trimmed = trim_history(_conv(10), budget=60)
    assert isinstance(trimmed[1], HumanMessage)


def test_trim_noop_under_budget():
    from app.services.chat_service import trim_history

    msgs = _conv(2)
    trimmed = trim_history(msgs, budget=100_000)
    assert len(trimmed) == len(msgs)


def test_to_langchain_messages_maps_roles():
    from app.schemas.chat import ChatMessage
    from app.services.chat_service import to_langchain_messages

    out = to_langchain_messages(
        [
            ChatMessage(role="user", content="你好"),
            ChatMessage(role="assistant", content="在的"),
        ]
    )
    assert isinstance(out[0], HumanMessage)
    assert isinstance(out[1], AIMessage)
