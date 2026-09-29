"""ch08 T6(integration):真 Streamable HTTP 链路——发现合并/经 MCP 查/拒连降级。
server 以子进程起,P9 手工验收 1/3 用同两条命令(README 收录)。"""
import socket
import subprocess
import sys
import time

import pytest

from app.core.config import Settings
from app.tools import executor
from app.tools.executor import ToolContext, execute_tool
from app.tools.registry import BUILTIN_SPECS, snapshot_tools

pytestmark = pytest.mark.integration


def _wait_port(host, port, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


@pytest.fixture
async def servers():
    procs = [subprocess.Popen([sys.executable, f"mcp_servers/{n}_server.py"])
             for n in ("logistics", "aftersale")]
    try:
        assert _wait_port("127.0.0.1", 8101) and _wait_port("127.0.0.1", 8102)
        yield
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=10)


async def test_mcp_discovery_call_chain(servers):
    st = Settings(openai_base_url="x", openai_api_key="x", model_name="x")
    specs = await snapshot_tools(st)
    assert specs["query_logistics"].source == "mcp"
    assert {"query_warranty", "query_return_progress"} <= set(specs)
    out = await execute_tool(specs["query_logistics"], {"order_id": "1001"}, "c1",
                             ToolContext(timeout_seconds=5))
    assert out.ok
    # server 回原码、executor 翻人话,二形态都认(格式化面 T5 已钉,这里只验链路)
    status = out.result["current_status"]
    assert status in executor.STATUS_LABELS.values() or \
        status in ("运输中", "派送中", "已签收", "已揽收")


async def test_server_down_snapshot_degrades():
    st = Settings(openai_base_url="x", openai_api_key="x", model_name="x",
                  mcp_logistics_url="http://127.0.0.1:9599/mcp",
                  mcp_aftersale_url="http://127.0.0.1:9598/mcp")
    specs = await snapshot_tools(st)          # 无 client 现构,端口无人听
    assert set(specs) == set(BUILTIN_SPECS)   # 不抛,内置面完整
