import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.runnables import Runnable


class FakeChunk:
    def __init__(self, text):
        self.text = text
        self.content = text


class FakeStructuredRunnable(Runnable):
    """with_structured_output 的返回值必须实现 Runnable 协议（Task 6 教训）。"""

    async def ainvoke(self, value, config=None, **kwargs):
        return FakeChatModel.structured_result

    def invoke(self, value, config=None, **kwargs):
        return FakeChatModel.structured_result


class FakeChatModel:
    """路由测试用假模型：流式吐固定文本，结构化提取返回 structured_result。"""

    structured_result = None  # 由各测试按需覆盖

    async def astream(self, messages, **kwargs):
        for t in ["你好", "呀", "喵"]:
            yield FakeChunk(t)

    def with_structured_output(self, schema, **kwargs):
        return FakeStructuredRunnable()


class BrokenChatModel:
    async def astream(self, messages, **kwargs):
        raise RuntimeError("upstream down")
        yield  # pragma: no cover （使其成为 async generator）

    def with_structured_output(self, schema, **kwargs):
        raise RuntimeError("upstream down")


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("MODEL_NAME", "fake-model")
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def client(fake_env):
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
