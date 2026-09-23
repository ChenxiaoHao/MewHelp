"""ch04 查询理解(spec §4.1):一次 LLM 调用 → {标准问法, ≤4 同义词};失败降级原话直检。

同义词只查询侧、不入库拆存(§0-10):dense 腿吃 standard_query,BM25 腿吃 bm25_text。
parse_rewrite 纯函数 TDD;prompt 改写质量走标注样例冒烟(evals/smoke_query_rewrite.py)。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from app.core.config import Settings
from app.prompts.query_rewrite import REWRITE_PROMPT

logger = logging.getLogger(__name__)

LLM_TIMEOUT_SECONDS = 15.0
MAX_SYNONYMS = 4
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.M)


@dataclass
class UnderstandResult:
    standard_query: str
    synonyms: list[str] = field(default_factory=list)
    degraded: bool = False

    @property
    def bm25_text(self) -> str:
        return " ".join([self.standard_query, *self.synonyms])


def parse_rewrite(raw) -> tuple[str, list[str]] | None:
    """dict/JSON 字符串皆可;不合 schema → None(调用方降级)。容忍 ```json 围栏。"""
    if isinstance(raw, str):
        try:
            data = json.loads(_FENCE.sub("", raw.strip()))
        except ValueError:
            return None
    else:
        data = raw
    if not isinstance(data, dict):
        return None
    std = data.get("standard_query")
    syns = data.get("synonyms")
    if not isinstance(std, str) or not std.strip() or not isinstance(syns, list):
        return None
    out: list[str] = []
    for s in syns:
        if isinstance(s, str) and (s := s.strip()) and s != std.strip() and s not in out:
            out.append(s)
        if len(out) >= MAX_SYNONYMS:
            break
    return std.strip(), out


async def understand_query(query: str, st: Settings, *, model=None) -> UnderstandResult:
    if not st.query_rewrite_enabled:
        return UnderstandResult(standard_query=query)
    try:
        from app.services.chat_service import get_model  # 延迟 import:rag 层不反向拖 services 依赖

        m = model or get_model(st)
        # 两段式等价于 REWRITE_PROMPT | m:installed langchain-core 1.x 的 `|`
        # 要求右端为 RunnableLike,会拒本模块测试用的最小替身(纯类,非 Runnable);
        # format_messages → ainvoke 与 RunnableSequence 末步行为一致,真模型无差异。
        messages = REWRITE_PROMPT.format_messages(question=query)
        resp = await asyncio.wait_for(m.ainvoke(messages), timeout=LLM_TIMEOUT_SECONDS)
        parsed = parse_rewrite(getattr(resp, "content", resp))
        if parsed is None:
            raise ValueError(f"rewrite 输出不合 schema: {str(resp)[:200]}")
        std, syn = parsed
        return UnderstandResult(standard_query=std, synonyms=syn)
    except Exception:  # noqa: BLE001 —— §8:理解层断了宁可退回原话,不许拖垮检索
        logger.warning("query 理解失败,原话直检: %s", query, exc_info=True)
        return UnderstandResult(standard_query=query, degraded=True)
