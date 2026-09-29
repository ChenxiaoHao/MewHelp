"""ch08 售后 MCP Server(需求6):独立进程,8102 /mcp,Streamable HTTP。
在保判定与退货进度均为 mock 播种;found=false 走正常返回(空结果不是异常,需求4)。"""
import random
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("aftersale", host="127.0.0.1", port=8102)


@mcp.tool()
async def query_warranty(order_id: str) -> dict:
    """按订单号查询是否在保修期:在保布尔、到期日与判定依据。order_id: 订单号。"""
    rnd = random.Random(f"warranty-{order_id}")
    days = rnd.randint(-60, 300)
    return {"order_id": order_id, "in_warranty": days > 0,
            "expire_date": (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d"),
            "basis": "自签收日起 365 天"}


@mcp.tool()
async def query_return_progress(order_id: str) -> dict:
    """按订单号查询退货单进度:无退货单返回 found=false。order_id: 订单号。"""
    if order_id.endswith("9"):                  # 演示可解释:尾号 9 暂无在途退货单
        return {"found": False, "order_id": order_id}
    rnd = random.Random(f"return-{order_id}")
    flow = ["APPROVED", "RECEIVING", "REFUNDING", "CLOSED"]
    i = rnd.randint(0, 3)
    return {"found": True, "order_id": order_id, "return_no": f"R{order_id}{i}",
            "current_status": flow[i],
            "nodes": [{"status": s, "desc": {"APPROVED": "审核通过",
                     "RECEIVING": "收到退货中", "REFUNDING": "退款处理中",
                     "CLOSED": "已关闭"}[s]} for s in flow[:i + 1]]}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
