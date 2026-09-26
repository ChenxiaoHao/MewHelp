import pytest

from app.core.config import get_settings


@pytest.mark.integration
async def test_ch05_upstream_model_tool_calls_and_json():
    """spec 开放核对项(需真 key):①bind_tools 后 tool_calls 命中且 args 为 dict
    ②JSON 指令输出可解析。Task 1 冒烟的一次性回归件。"""
    import json

    from langchain_core.messages import HumanMessage

    from app.services.chat_service import get_model
    from app.tools.registry import get_tools

    st = get_settings()
    bound = get_model(st).bind_tools(get_tools())
    ai = await bound.ainvoke([HumanMessage("订单 1001 的物流到哪了")])
    assert ai.tool_calls, "上游模型未产生 tool_calls——选型矛盾,停下问用户(D6)"
    assert all(isinstance(tc["args"], dict) for tc in ai.tool_calls)

    j = await get_model(st).ainvoke('只输出一个 JSON 对象 {"intent":"闲聊"},不要任何其他字符')
    assert json.loads(j.content) == {"intent": "闲聊"}
