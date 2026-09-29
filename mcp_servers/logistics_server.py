"""ch08 物流 MCP Server(需求6):独立进程,8101 /mcp,Streamable HTTP。
mock 数据照 ch02 做法按订单号播种(演示可复现),不接真实系统、不建表。"""
import random
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP   # T6 Step 0 定名:mcp1.30 载 v1 形制(ledger)

_CARRIERS = ["中通快递", "圆通速递", "韵达快递", "顺丰速运"]
_STATUS_FLOW = ["COLLECTED", "TRANSPORT", "TRANSPORT", "DELIVERING", "SIGNED"]

mcp = FastMCP("logistics", host="127.0.0.1", port=8101)


@mcp.tool()
async def query_logistics(order_id: str) -> dict:
    """按订单号查询物流轨迹:承运商、运单号、当前状态码与最近节点。

    order_id: 订单号,如 1001。
    """
    rnd = random.Random(f"logistics-{order_id}")
    n = rnd.randint(2, 4)
    now = datetime.now()
    picked = rnd.sample(_STATUS_FLOW, n)
    nodes = [{"time": (now - timedelta(hours=8 * (n - i))).strftime("%Y-%m-%d %H:%M"),
              "status": picked[i],
              "desc": {"COLLECTED": "快件已揽收", "TRANSPORT": "运输中",
                       "DELIVERING": "派送中", "SIGNED": "已签收"}[picked[i]],
              } for i in range(n)]
    return {"carrier": rnd.choice(_CARRIERS),
            "tracking_no": f"ZTO{rnd.randint(10**11, 10**12 - 1)}",
            "current_status": picked[-1],
            "nodes": nodes}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
