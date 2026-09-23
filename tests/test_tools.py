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


async def test_query_faq_contract_v2(monkeypatch):
    """契约 v2:refused/note 顶层键、hits 带 n 与 section_path、首尾排布生效、question=首行。"""
    from types import SimpleNamespace

    from app.rag.retriever import RetrieveResult, ScoredRow
    from app.tools import definitions as d

    def row(i):
        return SimpleNamespace(id=i, category="c", questions=f"q{i}a\nq{i}b", answer=f"a{i}",
                               section_path=f"手册 > 节{i}", content_type="policy")

    chunks = [ScoredRow(i, 1.0 - i / 10, row(i)) for i in range(1, 11)]

    async def fake_retrieve(query, **kw):
        assert query == "幼猫喂几次"
        return RetrieveResult(chunks=chunks)

    monkeypatch.setattr(d, "retrieve", fake_retrieve)
    out = await d.query_faq.ainvoke({"keyword": "幼猫喂几次"})
    assert out["refused"] is False and out["note"] == "" and len(out["hits"]) == 10
    assert [h["id"] for h in out["hits"]] == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]  # §4.4 排布
    assert [h["n"] for h in out["hits"]] == list(range(1, 11))
    assert out["hits"][0]["question"] == "q1a" and out["hits"][0]["section_path"] == "手册 > 节1"


async def test_query_faq_refused_pools_and_keeps_shape(monkeypatch):
    from app.rag.retriever import RetrieveResult
    from app.tools import definitions as d

    seen = {}

    async def fake_retrieve(query, **kw):
        return RetrieveResult(chunks=[], refused=True, note="证据置信度不足(top1=0.05 < 阈值 0.3)")

    async def fake_pool(cid, q, source, reason):
        seen.update(cid=cid, q=q, source=source)

    monkeypatch.setattr(d, "retrieve", fake_retrieve)
    monkeypatch.setattr(d, "pool_low_confidence", fake_pool)
    out = await d.query_faq.ainvoke({"keyword": "太空电梯门票"},
                                    config={"configurable": {"conversation_id": 42}})
    assert out == {"keyword": "太空电梯门票", "hits": [], "refused": True,
                   "note": "证据置信度不足(top1=0.05 < 阈值 0.3)"}
    assert seen == {"cid": 42, "q": "太空电梯门票", "source": "retrieval_low_conf"}


def test_query_faq_visible_args_still_only_keyword():
    """核对点④复核:config: RunnableConfig 不进模型可见 schema,签名兼容红线。"""
    from app.tools.registry import get_tool

    assert sorted(get_tool("query_faq").args.keys()) == ["keyword"]


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
