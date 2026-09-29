"""ch08 T2:注册中心合并/撞名丢弃/单 server 降级(spec 注册中心节)。"""
import pytest
from langchain_core.tools import tool

from app.core.config import Settings
from app.tools import registry


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"fake {name}"
        self.args = {}


class _BoomClient:
    async def get_tools(self, *, server_name):
        raise ConnectionError("server gone")


class _OneSidedClient:
    """logistics 给两件(其一撞内置),aftersale 给一件。"""
    async def get_tools(self, *, server_name):
        if server_name == "logistics":
            return [_FakeTool("query_waimai"), _FakeTool("query_order")]
        return [_FakeTool("query_warranty")]


def _settings():
    return Settings(openai_base_url="x", openai_api_key="x", model_name="x")


def test_builtin_specs_shape():
    assert list(registry.BUILTIN_SPECS) == ["query_order", "query_product",
                                            "query_faq", "create_ticket"]   # 无 logistics
    assert registry.BUILTIN_SPECS["create_ticket"].permission == "write"
    assert registry.BUILTIN_SPECS["query_faq"].permission == "readonly"
    assert all(s.source == "builtin" for s in registry.BUILTIN_SPECS.values())


async def test_snapshot_merges_mcp_and_drops_name_clash():
    specs = await registry.snapshot_tools(_settings(), client=_OneSidedClient())
    assert "query_waimai" in specs and "query_warranty" in specs
    assert specs["query_waimai"].source == "mcp"
    assert specs["query_waimai"].mcp_server == "logistics"
    assert specs["query_waimai"].permission == "readonly"   # M1-F2/P5:MCP 恒只读钉死
    assert specs["query_order"].source == "builtin"      # 撞名:内置存活


async def test_snapshot_single_server_failure_degrades(caplog):
    class Half(_OneSidedClient):
        async def get_tools(self, *, server_name):
            if server_name == "aftersale":
                raise ConnectionError("boom")
            return [_FakeTool("query_waimai")]

    specs = await registry.snapshot_tools(_settings(), client=Half())
    assert "query_waimai" in specs and "query_warranty" not in specs
    assert set(specs) >= set(registry.BUILTIN_SPECS)     # 内置面完整
    assert any("mcp discovery" in r.getMessage() for r in caplog.records)


async def test_snapshot_boom_client_all_servers_builtin_only():
    specs = await registry.snapshot_tools(_settings(), client=_BoomClient())
    assert set(specs) == set(registry.BUILTIN_SPECS)
