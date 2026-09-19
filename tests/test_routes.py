import json


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


async def test_chat_stream_sse_frames(client):
    from app.main import app
    from tests.conftest import FakeChatModel

    _override_model(app, FakeChatModel())
    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "event: token" in body
    # data 为 JSON 字符串化的增量文本
    token_lines = [
        line.removeprefix("data:").strip()
        for line in body.splitlines()
        if line.startswith("data:")
        and line.removeprefix("data:").strip() not in ("[DONE]",)
    ]
    joined = "".join(json.loads(t) for t in token_lines if t.startswith('"'))
    assert joined == "你好呀喵"
    assert "[DONE]" in body
    app.dependency_overrides.clear()


async def test_chat_stream_validation_last_not_user(client):
    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "assistant", "content": "嗨"}]},
    )
    assert r.status_code == 422


async def test_chat_stream_error_event_on_upstream_failure(client):
    from app.main import app
    from tests.conftest import BrokenChatModel

    _override_model(app, BrokenChatModel())
    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )
    assert r.status_code == 200  # SSE 惯例：错误走事件流
    assert "event: error" in r.text
    app.dependency_overrides.clear()


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
