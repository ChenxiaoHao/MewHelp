"""ch06 Task 5: POST /api/refunds(拍板 P6:落 tickets 表,ticket_type=「售后」)。

描述由服务端拼装(防注入面:模型/前端都碰不到文案);reason 固定四类,非法 422;
DB 未起 503。夹具模式与 test_tickets_endpoint_ch05 同源。
"""

from types import SimpleNamespace

from app.api import routes as routes_mod
from app.main import app


class FakeCrud:
    def __init__(self):
        self.calls = []

    async def create_ticket(self, session, *, conversation_id, description, ticket_type):
        self.calls.append({"cid": conversation_id, "desc": description, "type": ticket_type})
        return SimpleNamespace(ticket_no=f"R20260927{len(self.calls):03d}")


def _override_db(value):
    async def dep():
        yield value
    app.dependency_overrides[routes_mod.dep_db_session] = dep


async def test_refund_201_lands_ticket_aftersale(client, monkeypatch):
    fake = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override_db(object())
    r = await client.post("/api/refunds",
                          json={"conversation_id": 7, "order_id": "1001",
                                "reason": "七天无理由"})
    assert r.status_code == 201
    assert r.json() == {"ticket_no": "R20260927001"}
    call = fake.calls[0]
    assert call["type"] == "售后" and call["cid"] == 7
    assert "订单 1001" in call["desc"] and "七天无理由" in call["desc"]
    app.dependency_overrides.clear()


async def test_reason_enum_422(client, monkeypatch):
    fake = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override_db(object())
    r = await client.post("/api/refunds",
                          json={"conversation_id": 7, "order_id": "1001",
                                "reason": "想退就退"})
    assert r.status_code == 422
    assert fake.calls == []
    app.dependency_overrides.clear()


async def test_order_id_shape_422(client, monkeypatch):
    fake = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override_db(object())
    r = await client.post("/api/refunds",
                          json={"conversation_id": 7, "order_id": "abc-9999",
                                "reason": "其他"})
    assert r.status_code == 422
    app.dependency_overrides.clear()


async def test_order_id_fullwidth_digits_422(client, monkeypatch):
    # M3 评审回归:全角单号(\d 的 Unicode 穿透)不许进 tickets 表——
    # 脏票号无正则/种子可对齐,与 5712e93 槽位面同一口径。
    fake = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override_db(object())
    r = await client.post("/api/refunds",
                          json={"conversation_id": 7, "order_id": "１００２",
                                "reason": "七天无理由"})
    assert r.status_code == 422
    assert fake.calls == []
    app.dependency_overrides.clear()


async def test_no_db_503(client):
    async def none_session():
        yield None
    app.dependency_overrides[routes_mod.dep_db_session] = none_session
    r = await client.post("/api/refunds",
                          json={"conversation_id": 7, "order_id": "1001",
                                "reason": "商品质量问题"})
    assert r.status_code == 503
    app.dependency_overrides.clear()
