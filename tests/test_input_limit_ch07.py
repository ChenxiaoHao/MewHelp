"""ch07 T1 需求4-P5:当前句超 max_user_input_tokens → 422(拒,不截不500)。

校验器走 ChatRequest model_validator,估算口径与预算同源(estimate_text)。
monkeypatch get_settings 控制上限,避免测试依赖 .env 值。
"""

import pytest
from pydantic import ValidationError

import app.schemas.chat as chat_schema
from app.schemas.chat import ChatRequest


def _req(content):
    return {"messages": [{"role": "user", "content": content}]}


@pytest.fixture
def low_limit(monkeypatch):
    monkeypatch.setattr(chat_schema, "get_settings",
                        lambda: SimpleNs(max_user_input_tokens=50))


class SimpleNs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_over_limit_rejected(low_limit):
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(_req("汉" * 51))


def test_within_limit_passes(low_limit):
    ChatRequest.model_validate(_req("汉" * 40))


async def test_http_422_not_500(client):
    resp = await client.post("/api/chat/stream",
                             json=_req("退" * 2100))  # 默认上限2000 token → 超
    assert resp.status_code == 422
