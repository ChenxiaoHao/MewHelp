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


class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str          # 本轮用户原话
    resolved_query: str      # 指代消解输出（本章原样透传）
    intent: str              # 七类中文名之一
    route: str               # knowledge/data/complaint/chitchat
    evidence: list[dict]     # [{chunk_id, score, text}]
    gate_pass: bool          # 置信度闸结论
    suggestions: list[dict]  # [{action,label}] 转人工/建工单,互不绑定
    answer_text: str         # 最终答复（流式版在 Task 6 从事件帧出）
    log: dict                # {nodes:[...], retrieve_hits, agent_steps, ...}
