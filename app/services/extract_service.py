from app.prompts.extraction import EXTRACTION_PROMPT
from app.schemas.extraction import AfterSaleExtraction


async def extract_after_sale(description: str, model) -> AfterSaleExtraction:
    """售后描述 → 结构化字段（订单号/诉求类型/期望方案）。

    with_structured_output 用法已按 Context7 核对（langchain 1.x，2026-09-19）。
    """
    structured_model = model.with_structured_output(AfterSaleExtraction)
    chain = EXTRACTION_PROMPT | structured_model
    return await chain.ainvoke({"description": description})
