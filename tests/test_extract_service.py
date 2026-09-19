from langchain_core.runnables import Runnable


class FakeStructuredModel(Runnable):
    """模拟 with_structured_output 返回的 runnable（必须实现 Runnable 协议才能进 | 管道）。"""

    def __init__(self, schema, result):
        self.schema = schema
        self.result = result

    def invoke(self, value, config=None, **kwargs):
        # 管道上游是 EXTRACTION_PROMPT：本步收到的是渲染后的 PromptValue，不是原始 dict
        assert hasattr(value, "to_messages")
        return self.result

    async def ainvoke(self, value, config=None, **kwargs):
        assert hasattr(value, "to_messages")
        rendered = value.to_string()
        # description 已被渲染进消息（few-shot system + human 原文）
        assert "DD20260901001" in rendered
        return self.result


class FakeModelWithStructured:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema, **kwargs):
        return FakeStructuredModel(schema, self.result)


async def test_extract_returns_schema_instance():
    from app.schemas.extraction import AfterSaleExtraction, IssueType
    from app.services.extract_service import extract_after_sale

    expected = AfterSaleExtraction(
        order_id="DD20260901001",
        issue_type=IssueType.return_goods,
        expected_solution="退货并退款",
    )
    model = FakeModelWithStructured(expected)
    out = await extract_after_sale("耳机坏了要退货，订单DD20260901001", model)
    assert out == expected
