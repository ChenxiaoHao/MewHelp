"""ch07 T3:层2 半压渲染 + 降级切割(只挪 id) + ContextStore 读侧。

Review Focus 1 的形态红线:半压/回填后的消息序列**绝不出现 ToolMessage、绝不出现
带 tool_calls 的 AIMessage**——工具链(assistant+tool_calls 与其后连续 tool 行)
整体折叠成一条 marker AIMessage,否则 OpenAI 兼容端直接 400。
降级=边界 id 移动,不搬数据(spec「三层结构与边界语义」)。
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.context.budget import estimate_text
from app.db import crud


def _head(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "…"


def _fold_marker(tool_row, name: str) -> str:
    return f"[工具结果·{name}·≈{estimate_text(tool_row.content or '')} token 已折叠]"


def render_layer2(rows, settings) -> list[BaseMessage]:
    """三规则:user 原样 / assistant 留头几十字 / 工具链整链折一行标识。"""
    out: list[BaseMessage] = []
    i, n = 0, len(rows)
    while i < n:
        r = rows[i]
        if r.role == "user":
            out.append(HumanMessage(r.content or ""))
            i += 1
        elif r.role == "assistant" and r.tool_calls:
            parts = []
            if r.content:
                parts.append(_head(r.content, settings.assistant_head_chars))
            names = {
                tc.get("id"): tc.get("name", "?")
                for tc in r.tool_calls if isinstance(tc, dict)
            }
            j = i + 1
            while j < n and rows[j].role == "tool":
                parts.append(_fold_marker(rows[j], names.get(rows[j].tool_call_id, "?")))
                j += 1
            out.append(AIMessage("\n".join(parts)))
            i = j
        elif r.role == "assistant":
            out.append(AIMessage(_head(r.content or "", settings.assistant_head_chars)))
            i += 1
        else:  # 孤儿 tool 行(写路径理论不产生):折标识,绝不还原 ToolMessage
            out.append(AIMessage(_fold_marker(r, "?")))
            i += 1
    return out


def pick_degrade_cut(layer1_rows, budget, est) -> int | None:
    """返回新 layer1_from(降级批末行 id);None=不动(未超/无可降粒度)。

    从最新往老累计,纯预算可保留的起点若落在轮中(assistant/tool),向老对齐到
    该轮 user 行之前——一轮不拆成半压+原文两态;代价是保留段可略超预算(下轮
    自然再触发)。批首即 user 且整批是最小粒度时返回 None,防空转。
    """
    if not layer1_rows or est(layer1_rows) <= budget:
        return None
    keep_from = len(layer1_rows)
    while keep_from > 0 and est(layer1_rows[keep_from - 1:]) <= budget:
        keep_from -= 1
    k = max(keep_from - 1, 0)
    while k > 0 and layer1_rows[k].role != "user":
        k -= 1
    if k <= 0 or layer1_rows[k].role != "user":
        return None
    return layer1_rows[k - 1].id


class ContextStore:
    """DB 权威读侧薄包(crud 活库往返已由 test_crud_context_ch07 钉过,T7 接图消费)。"""

    def __init__(self, session_factory, conversation_id: int, settings):
        self._sf = session_factory
        self.cid = conversation_id
        self.settings = settings

    async def load_ctx(self) -> tuple[str | None, int, int]:
        async with self._sf() as session:
            return await crud.get_conv_ctx(session, self.cid)

    async def fetch_all_rows(self, limit: int = 10000):
        async with self._sf() as session:
            return await crud.list_messages_after(session, self.cid, 0, limit=limit)

    async def fetch_layer1(self, limit: int = 10000):
        _, _, layer1_from = await self.load_ctx()
        async with self._sf() as session:
            return await crud.list_messages_after(session, self.cid, layer1_from, limit=limit)

    async def set_layer1_from(self, value: int) -> None:
        async with self._sf() as session:
            await crud.set_layer1_from(session, self.cid, value)
