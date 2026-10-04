import json


async def test_persister_maps_hooks_to_rows(monkeypatch):
    """spec §7：on_tool_calls→assistant行(tool_calls JSON)；on_tool_result→tool行(结果JSON+tool_call_id)；on_final_answer→assistant行(全文)。"""
    from app.services import persistence as pers
    from app.tools.executor import ToolOutcome

    rows = []

    async def fake_add_message(
        session, conversation_id, role, content=None, tool_calls=None,
        tool_call_id=None, retrieval_snapshot=None  # ch09 T5 加参改钉
    ):
        rows.append((conversation_id, role, content, tool_calls, tool_call_id))

    monkeypatch.setattr(pers.crud, "add_message", fake_add_message)
    p = pers.DBChatPersister(session=object(), conversation_id=7)

    await p.on_tool_calls(7, "", [{"id": "c1", "name": "query_faq", "args": {"keyword": "退货"}}])
    outcome = ToolOutcome("query_faq", "c1", True, {"keyword": "退货", "hits": []}, "未命中")
    await p.on_tool_result(7, outcome)
    await p.on_final_answer(7, "最终回答")

    assert rows == [
        (7, "assistant", None, [{"id": "c1", "name": "query_faq", "args": {"keyword": "退货"}}], None),
        (7, "tool", json.dumps({"keyword": "退货", "hits": []}, ensure_ascii=False), None, "c1"),
        (7, "assistant", "最终回答", None, None),
    ]


async def test_persister_swallows_db_errors(monkeypatch):
    """运行期落库失败只 warning，绝不上抛（spec §9）。"""
    from app.services import persistence as pers
    from app.tools.executor import ToolOutcome

    async def boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(pers.crud, "add_message", boom)
    p = pers.DBChatPersister(session=object(), conversation_id=7)
    await p.on_tool_calls(7, "", [{"id": "c1", "name": "x", "args": {}}])
    await p.on_tool_result(7, ToolOutcome("x", "c1", False, {"error": "e"}, "失败"))
    await p.on_final_answer(7, "文本")  # 三个都不抛 = 通过
