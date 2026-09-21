import random

import pytest


def test_registry_contains_five_tools_in_order():
    from app.tools.registry import TOOL_REGISTRY, get_tool, get_tools

    assert list(TOOL_REGISTRY) == [
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    ]
    assert [t.name for t in get_tools()] == list(TOOL_REGISTRY)
    assert get_tool("query_order") is TOOL_REGISTRY["query_order"]
    assert get_tool("nope") is None


def test_create_ticket_schema_hides_conversation_id():
    """硬性核对点②的回归断言：模型可见参数只有 description/ticket_type。"""
    from app.tools.registry import get_tool

    assert sorted(get_tool("create_ticket").args.keys()) == [
        "description",
        "ticket_type",
    ]


def test_mock_tools_have_docstrings_and_schemas():
    from app.tools.registry import get_tool

    for name in ("query_order", "query_product", "query_logistics"):
        t = get_tool(name)
        assert t.description.strip(), f"{name} 缺 docstring（模型选工具靠它）"
        assert "order_id" in t.args or "product_id" in t.args


async def test_query_logistics_shape(monkeypatch):
    from app.tools import definitions as d

    monkeypatch.setattr(random, "randint", lambda a, b: a)  # 确定性
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    out = await d.query_logistics.ainvoke({"order_id": "1001"})
    assert out["order_id"] == "1001"
    assert out["carrier"] and out["current_status"]
    assert isinstance(out["traces"], list) and len(out["traces"]) >= 1
    assert {"time", "location", "detail"} <= set(out["traces"][0].keys())


async def test_query_order_and_product_shape():
    from app.tools import definitions as d

    o = await d.query_order.ainvoke({"order_id": "1001"})
    assert o["order_id"] == "1001"
    assert {"status", "amount", "items", "created_at"} <= set(o.keys())
    p = await d.query_product.ainvoke({"product_id": "2001"})
    assert p["product_id"] == "2001"
    assert {"name", "price", "stock", "category"} <= set(p.keys())


async def test_query_faq_hits(monkeypatch):
    """ch03 契约测试:返回键结构逐字段不变;question = questions 首行(多问法取首个)。"""
    from types import SimpleNamespace

    from app.tools import definitions as d

    row = SimpleNamespace(id=12, questions="幼猫一天喂几次\n小猫咪一天要吃几顿",
                          answer="每天 3-4 次。", category="猫粮")

    async def fake_retrieve(keyword, settings=None):
        assert keyword == "幼猫喂几次"
        return [row]

    monkeypatch.setattr(d, "retrieve_hits", fake_retrieve)
    out = await d.query_faq.ainvoke({"keyword": "幼猫喂几次"})
    assert out == {"keyword": "幼猫喂几次", "hits": [{"id": 12, "question": "幼猫一天喂几次",
                                                     "answer": "每天 3-4 次。", "category": "猫粮"}]}


async def test_query_faq_miss_shape(monkeypatch):
    """空命中返回结构化空数组,不回退 LIKE(§0-5 决策的测试化)。"""
    from app.tools import definitions as d

    async def fake_retrieve(keyword, settings=None):
        return []

    monkeypatch.setattr(d, "retrieve_hits", fake_retrieve)
    out = await d.query_faq.ainvoke({"keyword": "邮费"})
    assert out == {"keyword": "邮费", "hits": []}


async def test_create_ticket_uses_config_conversation_id(monkeypatch):
    from app.tools import definitions as d

    seen = {}

    async def fake_create_ticket(session, *, conversation_id, description, ticket_type):
        seen.update(conversation_id=conversation_id, description=description, ticket_type=ticket_type)
        from app.db.models import Ticket

        return Ticket(ticket_no="T20260920001", status="待处理")

    class Sess:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    # 注意：definitions.py 里 `from app.db.crud import create_ticket as crud_create_ticket`，
    # 打桩目标是模块内别名 crud_create_ticket，不是 @tool 对象本身
    monkeypatch.setattr(d, "crud_create_ticket", fake_create_ticket)
    monkeypatch.setattr(d, "get_session_factory", lambda: (lambda: Sess()))
    out = await d.create_ticket.ainvoke(
        {"description": "商品破损要退货", "ticket_type": "售后"},
        config={"configurable": {"conversation_id": 7}},
    )
    assert out == {"ticket_no": "T20260920001", "status": "待处理"}
    assert seen == {
        "conversation_id": 7,
        "description": "商品破损要退货",
        "ticket_type": "售后",
    }


async def test_create_ticket_without_conversation_raises(monkeypatch):
    """没有会话上下文（如评估脚本直调）→ ValueError，由 executor 包成错误结果。"""
    from app.tools import definitions as d

    class Sess:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(d, "get_session_factory", lambda: (lambda: Sess()))
    with pytest.raises(ValueError):
        await d.create_ticket.ainvoke(
            {"description": "x", "ticket_type": "咨询"},
            config={"configurable": {}},
        )
