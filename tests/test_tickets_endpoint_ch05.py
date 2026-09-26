"""ch05 Task 7: POST /api/tickets 端点单测（fake crud 转发/201/503/422）。

前端「建工单」按钮专用通道（需求 8：两按钮互不绑定）；DB 语义改动
（status 不动）归 test_create_ticket_decoupled_ch05 真库测。
"""

from types import SimpleNamespace

from app.api import routes as routes_mod
from app.main import app


class FakeCrud:
    def __init__(self):
        self.calls = []

    async def create_ticket(self, session, *, conversation_id, description, ticket_type):
        self.calls.append({"cid": conversation_id, "desc": description, "type": ticket_type})
        return SimpleNamespace(ticket_no=f"T20260926{len(self.calls):03d}")


def _override_db(value):
    async def dep():
        yield value
    app.dependency_overrides[routes_mod.dep_db_session] = dep


async def test_create_ticket_201_forwards_fields(client, monkeypatch):
    fake = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override_db(object())
    r = await client.post("/api/tickets",
                          json={"conversation_id": 7, "title": "物流卡了", "content": "三天没动"})
    assert r.status_code == 201
    assert r.json() == {"ticket_no": "T20260926001"}
    assert fake.calls[0]["cid"] == 7
    assert "物流卡了" in fake.calls[0]["desc"] and "三天没动" in fake.calls[0]["desc"]
    app.dependency_overrides.clear()


async def test_multiple_tickets_same_conversation_ok(client, monkeypatch):
    """重复会话多单合法：连击两次都 201，编号递进。"""
    fake = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override_db(object())
    body = {"conversation_id": 7, "title": "单A", "content": "内容A"}
    r1 = await client.post("/api/tickets", json=body)
    r2 = await client.post("/api/tickets", json=body)
    assert r1.status_code == r2.status_code == 201
    assert r1.json()["ticket_no"] != r2.json()["ticket_no"]
    app.dependency_overrides.clear()


async def test_no_db_503(client, monkeypatch):
    async def none_session():
        yield None
    app.dependency_overrides[routes_mod.dep_db_session] = none_session
    r = await client.post("/api/tickets",
                          json={"conversation_id": 7, "title": "t", "content": "c"})
    assert r.status_code == 503
    app.dependency_overrides.clear()


async def test_validation_422(client):
    r = await client.post("/api/tickets",
                          json={"conversation_id": 0, "title": "t", "content": "c"})
    assert r.status_code == 422
    r = await client.post("/api/tickets",
                          json={"conversation_id": 7, "title": "", "content": "c"})
    assert r.status_code == 422
    r = await client.post("/api/tickets", json={"conversation_id": 7, "title": "t"})
    assert r.status_code == 422
