def test_system_prompt_keeps_ch01_keywords():
    """ch01 test_prompts 的四个关键词必须保留（红线连带约束）。"""
    from app.prompts.customer_service import SYSTEM_PROMPT

    for kw in ["喵帮", "客服", "订单号", "转人工"]:
        assert kw in SYSTEM_PROMPT


def test_system_prompt_has_tool_guidance():
    """spec §10：工具指引三要素——只能来自工具 / 严禁编造 / 无结果时如实告知并建议工单。"""
    from app.prompts.customer_service import SYSTEM_PROMPT

    for kw in ["工具", "严禁编造", "人工工单"]:
        assert kw in SYSTEM_PROMPT


def test_ch01_no_system_claim_replaced():
    """ch01 约束 4 的『没有接入任何查询系统』能力声明必须已被工具指引取代（现在真的接了）。"""
    from app.prompts.customer_service import SYSTEM_PROMPT

    assert "没有接入任何订单/物流查询系统" not in SYSTEM_PROMPT
