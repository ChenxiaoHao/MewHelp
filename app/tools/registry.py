"""工具注册管理：唯一权威清单。bind_tools、executor 查找、评估脚本都从这里取，防止三处清单漂移。"""

from langchain_core.tools import BaseTool

from app.tools.definitions import (
    create_ticket,
    query_faq,
    query_logistics,
    query_order,
    query_product,
)

TOOL_REGISTRY: dict[str, BaseTool] = {
    "query_order": query_order,
    "query_product": query_product,
    "query_logistics": query_logistics,
    "query_faq": query_faq,
    "create_ticket": create_ticket,
}


def get_tools() -> list[BaseTool]:
    """bind_tools 用的工具列表（固定顺序，便于测试与排查）。"""
    return list(TOOL_REGISTRY.values())


def get_tool(name: str) -> BaseTool | None:
    return TOOL_REGISTRY.get(name)
