"""LangGraph 会话 State（ch05 spec「总体架构与图拓扑」节:State 一路贯穿）。

messages 用 add_messages reducer 累积（checkpointer 跨轮的地基）；
其余字段本轮内由对应节点写入,log.nodes 记录节点走过的顺序（验收1 依赖）。
本章不要求重启后恢复(D3):全部字段进程内有效。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# 操作建议常量（T5 起 nodes/agents 双向要用，落本无依赖模块断环）
TRANSFER_HUMAN = {"action": "transfer_human", "label": "转人工"}
CREATE_TICKET = {"action": "create_ticket", "label": "建工单"}
# ch06 需求 6/前端配套：退款闸通过后固定挂载的申请入口
REFUND_APPLY = {"action": "refund_apply", "label": "发起退款申请"}
# 订单选择器轮固定话术（cards 走 orders 帧,不进消息历史）
SELECT_ORDER_ASK = "好的，请从下方卡片选择要办理退款的订单："


class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str          # 本轮用户原话
    resolved_query: str      # ch06 真·指代消解+改写输出
    intent: str              # 八类中文名之一（ch06 起含「其他」）
    route: str               # knowledge/data/refund/complaint/chitchat
    evidence: list[dict]     # [{chunk_id, score, text}]
    gate_pass: bool          # 置信度闸结论
    suggestions: list[dict]  # [{action,label}] 转人工/建工单,互不绑定
    answer_text: str         # 最终答复（流式版在 Task 6 从事件帧出）
    log: dict                # {nodes:[...], retrieve_hits, agent_steps, ...}
    # --- ch06 新增（spec「State 扩展与逐轮复位」节；除 pending_flow 外均为轮级）---
    intent_confidence: float   # intent 节点判类置信度 0–1
    pending_flow: str          # "refund"=选择器已弹、下一轮 coref 读后即清
    slot_order_id: str         # 已确定的订单号（coref 续跑或 refund_slot 提取）
    order_data: dict           # refund_fetch 取到的订单详情（agent 注入用）
    expanded_queries: list[str]  # 实际参与政策检索的问法（≤4，原问法居首）
    orders_payload: list[dict]   # orders 帧数据（临时 UI，不落消息历史）
    # --- ch08 T7 建单确认流 ---
    ticket_preview: dict         # {tool_call_id, ticket_type, description}；非空=agent 后走确认节点
