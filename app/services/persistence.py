"""DBChatPersister：编排层 ChatPersister 协议（spec §7）的实现。

挂点对应关系：on_turn_start 在路由层直接走 crud（建/复用会话 + user 行，
发生在流开始前，需要拿到 conversation_id 才能装配后续一切）；其余三个挂点
由编排层在流中途回调。每个挂点内部自捕获异常只 warning——运行期落库失败
一律降级，绝不打断 SSE 聊天流（spec §9）。
"""

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.tools.executor import ToolOutcome

logger = logging.getLogger(__name__)


class DBChatPersister:
    def __init__(self, session: AsyncSession, conversation_id: int):
        self._session = session
        self._conversation_id = conversation_id

    async def on_tool_calls(self, conversation_id: int, content: str, tool_calls: list) -> None:
        try:
            await crud.add_message(
                self._session,
                self._conversation_id,
                "assistant",
                content=content or None,
                tool_calls=tool_calls,
            )
        except Exception:  # noqa: BLE001
            logger.warning("persist assistant(tool_calls) failed", exc_info=True)

    async def on_tool_result(self, conversation_id: int, outcome: ToolOutcome) -> None:
        try:
            await crud.add_message(
                self._session,
                self._conversation_id,
                "tool",
                content=json.dumps(outcome.result, ensure_ascii=False, default=str),
                tool_call_id=outcome.tool_call_id,
            )
        except Exception:  # noqa: BLE001
            logger.warning("persist tool result failed", exc_info=True)

    async def on_final_answer(self, conversation_id: int, content: str,
                              retrieval_snapshot: list | None = None) -> None:
        """ch09 T5:retrieval_snapshot 随行(👎 回捞数据源;闲聊/数据轮不传=NULL)。"""
        try:
            await crud.add_message(
                self._session, self._conversation_id, "assistant", content=content,
                retrieval_snapshot=retrieval_snapshot,
            )
        except Exception:  # noqa: BLE001
            logger.warning("persist final answer failed", exc_info=True)
