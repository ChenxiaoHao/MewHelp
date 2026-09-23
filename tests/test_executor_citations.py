"""ch04 executor 增量:citations 仅由 query_faq 命中派生,其余工具/空命中恒 None。

brief Files 清单点名本文件但步骤未给正文(计划欠账,dev-notes ④ 记录),
测试体按 Step 5 实现语义补齐:build_citations(result["hits"]) or None。
"""

from app.tools import executor as ex


class FakeTool:
    """替身:只需 ainvoke;get_tool 被打桩直接返回它。"""

    def __init__(self, name, result):
        self.name = name
        self._result = result

    async def ainvoke(self, args, config=None):
        return self._result


async def _run(monkeypatch, name, result):
    monkeypatch.setattr(ex, "get_tool", lambda n: FakeTool(name, result))
    return await ex.execute_tool(name, {"keyword": "退货政策"}, "c1", ex.ToolContext())


_HIT = {"n": 1, "id": 7, "question": "7 天无理由", "answer": "支持 7 天无理由退货。",
        "category": "退货政策", "section_path": "手册 > 退货"}


async def test_query_faq_hits_produce_citations(monkeypatch):
    out = await _run(monkeypatch, "query_faq",
                     {"keyword": "k", "hits": [_HIT], "refused": False, "note": ""})
    assert out.ok
    assert out.citations == [{"n": 1, "chunk_id": 7, "section_path": "手册 > 退货",
                              "question": "7 天无理由", "answer": "支持 7 天无理由退货。"}]


async def test_query_faq_empty_or_refused_citations_none(monkeypatch):
    out = await _run(monkeypatch, "query_faq",
                     {"keyword": "k", "hits": [], "refused": True, "note": "证据置信度不足"})
    assert out.ok and out.citations is None  # 空列表 or None → None(帧键缺席,契约红线)


async def test_legacy_hits_without_n_citations_none(monkeypatch):
    out = await _run(monkeypatch, "query_faq",
                     {"keyword": "k", "hits": [{"id": 1, "question": "旧", "answer": "旧", "category": "旧"}],
                      "refused": False, "note": ""})
    assert out.ok and out.citations is None  # build_citations 过滤无 n 旧形状 → 空 → None


async def test_other_tools_never_carry_citations(monkeypatch):
    out = await _run(monkeypatch, "query_order", {"order_id": "1001", "status": "待付款"})
    assert out.ok and out.citations is None


class BoomTool:
    name = "query_faq"

    async def ainvoke(self, args, config=None):
        raise RuntimeError("upstream down")


async def test_failure_outcome_citations_none(monkeypatch):
    monkeypatch.setattr(ex, "get_tool", lambda n: BoomTool())
    out = await ex.execute_tool("query_faq", {"keyword": "k"}, "c1",
                                ex.ToolContext(max_retries=0))
    assert not out.ok and out.citations is None
