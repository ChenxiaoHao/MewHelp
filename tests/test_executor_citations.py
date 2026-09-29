"""ch04 executor 增量:citations 仅由 query_faq 命中派生,其余工具/空命中恒 None。

ch08 T3:签名收 ToolSpec,直传假件(旧 get_tool 打桩模式作废)。
"""

from app.tools import executor as ex
from app.tools.registry import ToolSpec


class FakeTool:
    """替身:只需 ainvoke;由 ToolSpec 直传给 execute_tool。"""

    def __init__(self, name, result):
        self.name = name
        self.description = ""
        self.args = {}
        self._result = result

    async def ainvoke(self, args, config=None):
        return self._result


async def _run(name, result):
    return await ex.execute_tool(ToolSpec(FakeTool(name, result), "readonly", "builtin"),
                                 {"keyword": "退货政策"}, "c1", ex.ToolContext())


_HIT = {"n": 1, "id": 7, "question": "7 天无理由", "answer": "支持 7 天无理由退货。",
        "category": "退货政策", "section_path": "手册 > 退货"}


async def test_query_faq_hits_produce_citations():
    out = await _run("query_faq",
                     {"keyword": "k", "hits": [_HIT], "refused": False, "note": ""})
    assert out.ok
    assert out.citations == [{"n": 1, "chunk_id": 7, "section_path": "手册 > 退货",
                              "question": "7 天无理由", "answer": "支持 7 天无理由退货。"}]


async def test_query_faq_empty_or_refused_citations_none():
    out = await _run("query_faq",
                     {"keyword": "k", "hits": [], "refused": True, "note": "证据置信度不足"})
    assert out.ok and out.citations is None  # 空列表 or None → None(帧键缺席,契约红线)


async def test_legacy_hits_without_n_citations_none():
    out = await _run("query_faq",
                     {"keyword": "k", "hits": [{"id": 1, "question": "旧", "answer": "旧", "category": "旧"}],
                      "refused": False, "note": ""})
    assert out.ok and out.citations is None  # build_citations 过滤无 n 旧形状 → 空 → None


async def test_other_tools_never_carry_citations():
    out = await _run("query_order", {"order_id": "1001", "status": "待付款"})
    assert out.ok and out.citations is None


class BoomTool:
    name = "query_faq"
    description = ""
    args = {}

    async def ainvoke(self, args, config=None):
        raise RuntimeError("upstream down")


async def test_failure_outcome_citations_none():
    out = await ex.execute_tool(ToolSpec(BoomTool(), "readonly", "builtin"),
                                {"keyword": "k"}, "c1", ex.ToolContext(max_retries=0))
    assert not out.ok and out.citations is None
