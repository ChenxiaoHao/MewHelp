import pytest
from langchain_core.messages import HumanMessage


class FakeChunk:
    def __init__(self, text):
        self.text = text
        self.content = text  # 兼容两种属性取法


class FakeModel:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or ["你", "好", "喵"]
        self.error = error

    async def astream(self, messages, **kwargs):
        if self.error:
            raise self.error
        for t in self.chunks:
            yield FakeChunk(t)


async def test_stream_chat_yields_text_chunks():
    from app.services.chat_service import stream_chat

    out = [t async for t in stream_chat([HumanMessage(content="hi")], FakeModel())]
    assert out == ["你", "好", "喵"]


async def test_stream_chat_propagates_upstream_error():
    from app.services.chat_service import stream_chat

    model = FakeModel(error=RuntimeError("upstream down"))
    with pytest.raises(RuntimeError):
        _ = [t async for t in stream_chat([HumanMessage(content="hi")], model)]


def test_build_messages_prepends_system_and_trims(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv("HISTORY_TOKEN_BUDGET", "60")
    from app.core.config import Settings, get_settings
    from app.schemas.chat import ChatMessage
    from app.services.chat_service import build_messages

    get_settings.cache_clear()
    settings = Settings(_env_file=None)
    msgs = [
        ChatMessage(role="user", content=f"很长的问题{i}" * 30) for i in range(6)
    ]
    out = build_messages(msgs, settings)
    assert out[0].type == "system"
    assert out[-1].content == msgs[-1].content
    assert len(out) < len(msgs) + 1


def test_get_model_uses_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    monkeypatch.setenv("MODEL_NAME", "qwen2.5:7b")
    from app.core.config import Settings
    from app.services.chat_service import get_model

    m = get_model(Settings(_env_file=None))
    assert m.model_name == "qwen2.5:7b"
    assert m.temperature == 0.7
