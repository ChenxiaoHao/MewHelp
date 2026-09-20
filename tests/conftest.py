import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import Runnable


class FakeChunk:
    # ch02 起 conftest 不再使用它（FakeChatModel 改吐真实 AIMessageChunk）；
    # test_stream_chat.py 用的是自己的本地副本，不受影响。
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
    """路由测试用假模型：流式吐固定文本（真实 AIMessageChunk，支持 ch02 编排的
    chunk 累加/tool_calls 聚合），结构化提取返回 structured_result。"""

    structured_result = None  # 由各测试按需覆盖

    def __init__(self):
        self.bound_tools = None

    def bind_tools(self, tools, **kwargs):
        """ch02 编排会调用；记录工具清单并返回自身（流脚本已确定）。"""
        self.bound_tools = tools
        return self

    async def astream(self, messages, **kwargs):
        for t in ["你好", "呀", "喵"]:
            yield AIMessageChunk(content=t)

    def with_structured_output(self, schema, **kwargs):
        return FakeStructuredRunnable()


class BrokenChatModel:
    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages, **kwargs):
        raise RuntimeError("upstream down")
        yield  # pragma: no cover （使其成为 async generator）

    def with_structured_output(self, schema, **kwargs):
        raise RuntimeError("upstream down")


@pytest.fixture
def fake_settings():
    """显式构造的测试配置（init 参数优先级最高，不受 .env / 系统环境变量影响）。"""
    from app.core.config import Settings

    return Settings(
        _env_file=None,
        openai_base_url="http://fake/v1",
        openai_api_key="fake-key",
        model_name="fake-model",
        history_token_budget=4000,
        temperature=0.7,
    )


@pytest.fixture
async def client(fake_settings):
    from app.api.routes import dep_settings
    from app.main import app

    app.dependency_overrides[dep_settings] = lambda: fake_settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
