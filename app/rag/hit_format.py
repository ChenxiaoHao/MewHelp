"""命中组装纯函数(spec §5.1):query_faq / T12 评估 runner 共用一份,契约单源。"""

from __future__ import annotations

from typing import Any


def format_hits(chunks) -> list[dict]:
    """ScoredRow 列表(已过首尾排布)→ hits v2;n = 列表 1-based 位置(§4.4 [n] 语义)。"""
    return [
        {"n": i + 1, "id": c.chunk_id, "question": c.row.questions.splitlines()[0],
         "answer": c.row.answer, "category": c.row.category,
         "section_path": c.row.section_path or ""}
        for i, c in enumerate(chunks)
    ]


def build_citations(hits: list[dict]) -> list[dict]:
    """hits → citations(台账与原文弹窗回放集,answer 全文随带,附录 A 注)。"""
    return [
        {"n": h["n"], "chunk_id": h["id"], "section_path": h.get("section_path", ""),
         "question": h.get("question", ""), "answer": h.get("answer", "")}
        for h in hits if "n" in h
    ]
