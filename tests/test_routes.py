def _override_model(app, model):
    from app.api.routes import dep_chat_model

    app.dependency_overrides[dep_chat_model] = lambda: model
    return app


async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "status": "ok",
        "model": "fake-model",
        "history_token_budget": 4000,
    }


def _msgs(text):
    from app.schemas.chat import ChatMessage

    return [ChatMessage(role="user", content=text)]


async def test_chat_stream_sse_frames(fake_settings):
    """P5 基线:ch04 编排服务层 token 流。端点接线 ch05 起换 Graph,
    帧面/降级/error 回归归 tests/test_chat_stream_ch05.py。"""
    from app.services.tool_chat_service import stream_chat_with_tools
    from tests.conftest import FakeChatModel

    events = [
        ev async for ev in stream_chat_with_tools(_msgs("你好"), fake_settings, FakeChatModel())
    ]
    assert {k for k, _ in events} == {"token"}
    assert "".join(d for k, d in events if k == "token") == "你好呀喵"


async def test_chat_stream_validation_last_not_user(client):
    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "assistant", "content": "嗨"}]},
    )
    assert r.status_code == 422


async def test_chat_stream_error_event_on_upstream_failure(fake_settings):
    """P5 基线:服务层原样抛出(捕获转 error 帧是路由层职责,ch05 测已覆盖)。"""
    import pytest

    from app.services.tool_chat_service import stream_chat_with_tools
    from tests.conftest import BrokenChatModel

    with pytest.raises(RuntimeError, match="upstream down"):
        async for _ in stream_chat_with_tools(_msgs("你好"), fake_settings, BrokenChatModel()):
            pass


async def test_extract_returns_json(client):
    from app.main import app
    from app.schemas.extraction import AfterSaleExtraction, IssueType
    from tests.conftest import FakeChatModel

    FakeChatModel.structured_result = AfterSaleExtraction(
        order_id="DD123",
        issue_type=IssueType.refund,
        expected_solution="全额退款",
    )
    _override_model(app, FakeChatModel())
    r = await client.post(
        "/api/extract", json={"description": "订单DD123不想要了，退款"}
    )
    assert r.status_code == 200
    assert r.json() == {
        "order_id": "DD123",
        "issue_type": "退款",
        "expected_solution": "全额退款",
    }
    app.dependency_overrides.clear()


async def test_extract_502_on_upstream_failure(client):
    from app.main import app
    from tests.conftest import BrokenChatModel

    _override_model(app, BrokenChatModel())
    r = await client.post("/api/extract", json={"description": "任意描述"})
    assert r.status_code == 502
    assert "detail" in r.json()
    app.dependency_overrides.clear()


async def test_extract_rejects_empty_description(client):
    r = await client.post("/api/extract", json={"description": ""})
    assert r.status_code == 422
