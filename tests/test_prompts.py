from langchain_core.messages import AIMessage, HumanMessage


def test_customer_service_prompt_renders_with_system():
    from app.prompts.customer_service import CUSTOMER_SERVICE_PROMPT

    rendered = CUSTOMER_SERVICE_PROMPT.invoke(
        {"messages": [HumanMessage(content="你好")]}
    ).to_messages()
    assert rendered[0].type == "system"
    assert rendered[-1].content == "你好"


def test_customer_service_prompt_contains_constraints():
    from app.prompts.customer_service import CUSTOMER_SERVICE_PROMPT

    rendered = CUSTOMER_SERVICE_PROMPT.invoke(
        {
            "messages": [
                HumanMessage(content="在吗"),
                AIMessage(content="在的"),
                HumanMessage(content="怎么退货"),
            ]
        }
    ).to_messages()
    system_text = rendered[0].content
    # 角色设定与行为约束关键词
    for keyword in ["喵帮", "客服", "订单号", "转人工"]:
        assert keyword in system_text
    # 历史消息保序传入
    assert [m.type for m in rendered[1:]] == ["human", "ai", "human"]


def test_extraction_prompt_renders_description():
    from app.prompts.extraction import EXTRACTION_PROMPT

    rendered = EXTRACTION_PROMPT.invoke(
        {"description": "订单DD1的鞋子开胶了，想退货"}
    ).to_messages()
    assert rendered[-1].content == "订单DD1的鞋子开胶了，想退货"
    # few-shot 样例存在于 system 部分
    system_text = rendered[0].content
    assert "DD20260901001" in system_text
