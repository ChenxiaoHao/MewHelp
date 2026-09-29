"""工具注册中心(spec 注册中心节):三件套统一登记,内置启动登记,MCP 每轮现拿。

ToolSpec.permission 是权限唯一权威——外部 MCP 自带声明一概不采信(需求3);
snapshot_tools 每轮调用(P4 无缓存,adapters 实证每次 get_tools 新建会话),
单 server 连不上只 WARN 降级,聊天不断线(T7 store 面全吞原则延伸)。
"""

import logging
from dataclasses import dataclass
from typing import Literal

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.tools.definitions import create_ticket, query_faq, query_order, query_product

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    tool: BaseTool
    permission: Literal["readonly", "write"]   # 只认我方登记(P5)
    source: Literal["builtin", "mcp"]
    mcp_server: str | None = None

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def description(self) -> str:
        return self.tool.description or ""


BUILTIN_SPECS: dict[str, ToolSpec] = {
    "query_order": ToolSpec(query_order, "readonly", "builtin"),
    "query_product": ToolSpec(query_product, "readonly", "builtin"),
    "query_faq": ToolSpec(query_faq, "readonly", "builtin"),
    # 全系统唯一 write(需求3);执行闸在 executor 权限闸,bind 与否在 T7 切换
    "create_ticket": ToolSpec(create_ticket, "write", "builtin"),
}

_MCP_SERVERS = ("logistics", "aftersale")


def get_tools() -> list[BaseTool]:
    """内置视图(bind/legacy 面兼容签名;主力图链请用 snapshot_tools)。"""
    return [s.tool for s in BUILTIN_SPECS.values()]


def get_tool(name: str) -> BaseTool | None:
    spec = BUILTIN_SPECS.get(name)
    return spec.tool if spec else None


def make_mcp_client(settings) -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "logistics": {"transport": "http", "url": settings.mcp_logistics_url},
        "aftersale": {"transport": "http", "url": settings.mcp_aftersale_url},
    })


async def snapshot_tools(settings, *, client=None) -> dict[str, ToolSpec]:
    """当轮工具面:内置 ∪ 各 MCP server 现拿(server_name 逐连,失败只丢该面)。"""
    specs = dict(BUILTIN_SPECS)
    client = client or make_mcp_client(settings)
    for sname in _MCP_SERVERS:
        try:
            tools = await client.get_tools(server_name=sname)
        except Exception:  # noqa: BLE001 —— 发现面降级不断线
            logger.warning("mcp discovery failed server=%s; builtin view kept",
                           sname, exc_info=True)
            continue
        for t in tools:
            if t.name in specs:          # 撞名:内置权威,外部件丢弃(权限面不可污染)
                logger.warning("mcp tool %s (server=%s) clashes with builtin; dropped",
                               t.name, sname)
                continue
            specs[t.name] = ToolSpec(t, "readonly", "mcp", sname)   # P5:MCP 恒只读
    return specs
