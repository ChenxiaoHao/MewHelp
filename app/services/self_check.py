"""闸2:生成前证据自评(spec §5.2)。失败/超时 → None = 放行(§0-2 用户确认的降级方向)。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from app.core.config import Settings
from app.prompts.self_check import SELF_CHECK_PROMPT

logger = logging.getLogger(__name__)

EVIDENCE_TIMEOUT_SECONDS = 15.0
_ANSWER_EXCERPT = 200


class EvidenceCheck(BaseModel):
    sufficient: bool
    reason: str = ""


def _render_hits(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        ans = (h.get("answer") or "").replace("\n", " ")[:_ANSWER_EXCERPT]
        lines.append(f"[{h.get('n')}] {h.get('question', '')} | {h.get('section_path', '')} | {ans}")
    return "\n".join(lines)


async def evaluate_evidence(question: str, hits: list[dict], settings: Settings,
                            *, model: Any | None = None) -> EvidenceCheck | None:
    try:
        from app.services.chat_service import get_model  # 延迟 import,同 query_understanding 惯例

        m = model or get_model(settings)
        chain = SELF_CHECK_PROMPT | m.with_structured_output(EvidenceCheck)
        result = await asyncio.wait_for(
            chain.ainvoke({"question": question, "evidence": _render_hits(hits)}),
            timeout=EVIDENCE_TIMEOUT_SECONDS,
        )
        return EvidenceCheck.model_validate(result)
    except Exception:  # noqa: BLE001 —— §0-2:自评挂了不许拦生成
        logger.warning("证据自评失败,放行生成: %s", question, exc_info=True)
        return None
