"""ch07 T9: 只读会话 API×2(spec「HTTP API(P8 口径)」)。

GET /api/conversations = id 降序 + 首问预览截 40 字 + 已摘要标记,demo_user_id 面;
GET /api/conversations/{id}/messages = 全行升序含 tool 行,不存在/非属主 404,
引擎 None 503。夹具模式与 test_api_refunds_ch06 同源(monkeypatch routes.crud +
dependency_overrides dep_db_session),零真库。
"""

from datetime import datetime
from types import SimpleNamespace

from app.api import routes as routes_mod
from app.main import app


class FakeCrud:
    def __init__(self, rows=None, owned=True):
        self.calls = []
        self.rows = rows or []
        self.owned = owned

    async def list_user_conversations(self, session, user_id, limit=50):
        self.calls.append(("list", user_id, limit))
        return self.rows

    async def get_conversation_for_user(self, session, cid, user_id):
        self.calls.append(("own", cid, user_id))
        return SimpleNamespace(id=cid) if self.owned else None

    async def list_messages_after(self, session, cid, after_id, limit=500):
        self.calls.append(("msgs", cid, after_id, limit))
        return self.rows


def _override_db(value):
    async def dep():
        yield value
    app.dependency_overrides[routes_mod.dep_db_session] = dep


_ROW7 = (7, datetime(2026, 9, 28, 10, 0, 0), "退货政策是什么呀帮我看看这单能不能退", True)


async def test_list_conversations_maps_items_preview_cap40_summarized(client, monkeypatch):
    fake = FakeCrud(rows=[_ROW7])
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override_db(object())
    r = await client.get("/api/conversations")
    assert r.status_code == 200
    item = r.json()["items"][0]
    assert set(item) == {"id", "created_at", "preview", "summarized"}
    assert item["id"] == 7 and item["summarized"] is True
    assert item["preview"] == _ROW7[2]           # 原样透传:截 40 字是 crud 面(其 T2 测已钉),API 不二次加工
    assert fake.calls == [("list", "demo_user", 50)]   # demo_user_id 面,limit 走默认
    app.dependency_overrides.clear()


async def test_list_conversations_no_db_503(client):
    _override_db(None)
    r = await client.get("/api/conversations")
    assert r.status_code == 503
    app.dependency_overrides.clear()


async def test_messages_unknown_or_foreign_404_and_zero_read(client, monkeypatch):
    fake = FakeCrud(owned=False)
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override_db(object())
    r = await client.get("/api/conversations/999/messages")
    assert r.status_code == 404
    assert fake.calls == [("own", 999, "demo_user")]   # 归属未过=消息查询根本不发起
    app.dependency_overrides.clear()


async def test_messages_owner_200_ascending_with_tool_rows(client, monkeypatch):
    rows = [
        SimpleNamespace(id=3, role="tool", content='{"ok":true}', tool_calls=None,
                        created_at=datetime(2026, 9, 28, 10, 2)),
        SimpleNamespace(id=2, role="assistant", content="可以退", tool_calls=None,
                        created_at=datetime(2026, 9, 28, 10, 1)),
        SimpleNamespace(id=1, role="user", content="这单能退吗",
                        tool_calls=[{"id": "c1", "name": "query_order", "args": {}}],
                        created_at=datetime(2026, 9, 28, 10, 0)),
    ]
    fake = FakeCrud(rows=rows)
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override_db(object())
    r = await client.get("/api/conversations/7/messages")
    assert r.status_code == 200
    items = r.json()
    assert [m["id"] for m in items] == [3, 2, 1]        # crud 原序透传(升序由 crud 保证)
    assert set(items[0]) == {"id", "role", "content", "tool_calls", "created_at"}
    assert items[0]["role"] == "tool"                   # tool 行进回载(P8 口径)
    assert fake.calls == [("own", 7, "demo_user"), ("msgs", 7, 0, 10000)]  # I2 同律:显式大 limit
    app.dependency_overrides.clear()


async def test_messages_no_db_503(client):
    _override_db(None)
    r = await client.get("/api/conversations/7/messages")
    assert r.status_code == 503
    app.dependency_overrides.clear()
