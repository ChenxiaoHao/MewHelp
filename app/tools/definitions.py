"""五个业务工具。query_order/query_product/query_logistics 为 mock（不接真实接口、不建表）；
query_faq 走向量语义检索(app/rag/retriever.py,ch03);create_ticket 写 tickets 表。

工具函数一律返回 dict（结构化结果），异常向上抛由 executor 统一包装——
definitions 里不写错误处理（职责分离）。
"""

import random
from datetime import datetime, timedelta
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.db.crud import create_ticket as crud_create_ticket
from app.db.engine import get_session_factory
from app.rag.retriever import retrieve_hits

_ORDER_STATUSES = ["待付款", "待发货", "运输中", "已签收", "已取消"]
_CARRIERS = ["中通快递", "圆通速递", "韵达快递", "顺丰速运"]
_CITIES = ["杭州转运中心", "苏州分拨中心", "南京集散中心", "上海虹桥网点", "北京大兴网点"]
_PRODUCT_NAMES = ["喵帮定制猫爬架", "冻干鸡肉猫粮 2kg", "宠物自动饮水机", "猫砂盆除臭剂", "磨爪逗猫棒套装"]
_CATEGORIES = ["猫粮", "用品", "零食", "清洁"]


@tool
async def query_order(order_id: str) -> dict:
    """按订单号查询订单状态、金额与商品明细。用户问「我的订单」「订单到哪一步了」「订单状态」时使用。order_id: 订单号，如 1001。"""
    rnd = random.Random(f"order-{order_id}")  # 同订单号结果稳定，演示可复现
    n_items = rnd.randint(1, 3)
    return {
        "order_id": order_id,
        "status": rnd.choice(_ORDER_STATUSES),
        "amount": round(rnd.uniform(29, 599), 2),
        "created_at": (datetime.now() - timedelta(days=rnd.randint(0, 30))).strftime("%Y-%m-%d %H:%M"),
        "items": [
            {"name": rnd.choice(_PRODUCT_NAMES), "qty": rnd.randint(1, 2)}
            for _ in range(n_items)
        ],
    }


@tool
async def query_product(product_id: str) -> dict:
    """按商品编号查询商品名称、价格、库存与分类。用户问某个商品「多少钱」「有没有货」「商品信息」时使用。product_id: 商品编号，如 2001。"""
    rnd = random.Random(f"product-{product_id}")
    return {
        "product_id": product_id,
        "name": rnd.choice(_PRODUCT_NAMES),
        "price": round(rnd.uniform(9.9, 399.0), 2),
        "stock": rnd.randint(0, 200),
        "category": rnd.choice(_CATEGORIES),
    }


@tool
async def query_logistics(order_id: str) -> dict:
    """按订单号查询物流轨迹：承运商、当前状态与最近几条轨迹。用户问「物流到哪了」「快递走到哪了」「什么时候到」时使用。order_id: 订单号，如 1001。"""
    rnd = random.Random(f"logistics-{order_id}")
    n = rnd.randint(2, 4)
    now = datetime.now()
    cities = rnd.sample(_CITIES, n)
    traces = [
        {
            "time": (now - timedelta(hours=8 * (n - i))).strftime("%Y-%m-%d %H:%M"),
            "location": city,
            "detail": rnd.choice(["快件已到达", "快件已发出，下一站", "运输中", "已揽收"]),
        }
        for i, city in enumerate(cities)
    ]
    return {
        "order_id": order_id,
        "carrier": rnd.choice(_CARRIERS),
        "current_status": rnd.choice(["运输中", "派送中", "已签收", "已揽收"]),
        "traces": traces,
    }


@tool
async def query_faq(keyword: str) -> dict:
    """语义检索平台知识库,回答规则、政策、费用与商品使用类问题(退换货政策、运费与包邮门槛、售后流程、积分等)。用户咨询任何平台规则、政策、费用、商品用法类问题时,必须先调用本工具再作答,即使你认为自己知道通用答案。keyword: 用户的原始问题完整句子(语义检索按整句匹配,请勿自行拆词)。"""
    hits = await retrieve_hits(keyword)
    return {
        "keyword": keyword,
        "hits": [
            {"id": c.id, "question": c.questions.splitlines()[0], "answer": c.answer, "category": c.category}
            for c in hits
        ],
    }


@tool
async def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    config: RunnableConfig,
) -> dict:
    """创建人工客服工单（转人工）。仅当用户明确要求转人工，或问题超出工具与 FAQ 能力、需要人工跟进时使用。description: 用一句话概括用户的问题与诉求; ticket_type: 工单类型，售后/投诉/咨询三选一。"""
    conversation_id = (config.get("configurable") or {}).get("conversation_id")
    async with get_session_factory()() as session:
        ticket = await crud_create_ticket(
            session,
            conversation_id=conversation_id,
            description=description,
            ticket_type=ticket_type,
        )
        return {"ticket_no": ticket.ticket_no, "status": ticket.status}
