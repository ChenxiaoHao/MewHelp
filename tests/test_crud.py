from datetime import date


class FakeSession:
    """最小 AsyncSession 替身：只覆盖 crud 用到的方法。"""

    def __init__(self, existing=None):
        self.existing = existing
        self.added = []
        self.commits = 0
        self.rollbacks = 0
        self.refreshed = []

    async def get(self, model, pk):
        return self.existing

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1
        for o in self.added:
            if hasattr(o, "id") and getattr(o, "id") is None:
                o.id = 42  # 模拟自增主键回填

    async def rollback(self):
        self.rollbacks += 1

    async def refresh(self, obj):
        self.refreshed.append(obj)


def test_next_ticket_no_format():
    from app.db.crud import next_ticket_no

    d = date(2026, 9, 20)
    assert next_ticket_no(0, d) == "T20260920001"
    assert next_ticket_no(7, d) == "T20260920008"
    assert next_ticket_no(99, d) == "T20260920100"


def test_build_faq_query_is_or_like():
    from app.db.crud import build_faq_query

    sql = str(
        build_faq_query("退货", limit=3).compile(compile_kwargs={"literal_binds": True})
    )
    assert "question LIKE" in sql
    assert "answer LIKE" in sql
    assert "OR" in sql
    assert "%退货%" in sql
    assert "LIMIT" in sql


async def test_create_conversation_when_id_none():
    from app.db import crud

    s = FakeSession(existing=None)
    conv = await crud.create_or_get_conversation(s, None, "demo_user")
    assert conv.user_id == "demo_user"
    assert s.commits == 1
    assert conv in s.refreshed


async def test_reuse_existing_conversation():
    from app.db import crud
    from app.db.models import Conversation

    existing = Conversation(id=7, user_id="demo_user")
    s = FakeSession(existing=existing)
    conv = await crud.create_or_get_conversation(s, 7, "demo_user")
    assert conv is existing
    assert s.commits == 0


async def test_fallback_new_when_stale_id():
    """前端带了一个库里不存在的 conversation_id → 新建而不是报错。"""
    from app.db import crud

    s = FakeSession(existing=None)
    conv = await crud.create_or_get_conversation(s, 999, "demo_user")
    assert conv.id is not None or conv in s.refreshed
    assert s.commits == 1


async def test_add_message_passes_through_fields():
    from app.db import crud

    s = FakeSession()
    msg = await crud.add_message(
        s, 7, "assistant", content=None, tool_calls=[{"id": "c1", "name": "query_order"}]
    )
    assert msg.role == "assistant"
    assert msg.conversation_id == 7
    assert msg.tool_calls[0]["name"] == "query_order"
    assert s.commits == 1
