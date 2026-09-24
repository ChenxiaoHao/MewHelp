"""老师 300 题评估 CSV 的解析器与指标纯函数(spec §6.1/附录 B;evals/run_rag.py 是只读数据文件)。

期望章节语法:「+」=AND 组、「|」=组内 any-of、「 / 」≡「 > 」=路径层级;
原子按 section_path **子串**匹配。D 桶期望列空、应拒答=是:不参与排序指标,
只进闸1阈值校准(数字在 run_strategy_eval,T12 终值回写 Settings)。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

CSV_PATH = Path("evals/run_rag.py")  # 扩展名与内容不符(实为 UTF-8 CSV)——盘点结论,附录 B


@dataclass
class EvalQuestion:
    id: str
    bucket: str
    query: str
    groups: list[list[str]] = field(default_factory=list)  # AND 组,组内 OR 原子
    expect_points: str = ""
    should_refuse: bool = False


def parse_expect_section(raw) -> list[list[str]]:
    raw = (raw or "").strip()
    if not raw:
        return []
    groups: list[list[str]] = []
    for g in raw.split("+"):
        atoms = [a.replace(" / ", " > ").strip() for a in g.split("|") if a.strip()]
        if atoms:
            groups.append(atoms)
    return groups


def load_questions(path: Path = CSV_PATH) -> list[EvalQuestion]:
    with path.open(encoding="utf-8", newline="") as f:
        return [
            EvalQuestion(
                id=row["id"].strip(),
                bucket=row["桶(bucket)"].strip(),
                query=row["问题(query)"].strip(),
                groups=parse_expect_section(row["期望章节(expect_section)"]),
                expect_points=row["标准要点(expect_points)"].strip(),
                should_refuse=row["应拒答(should_refuse)"].strip() == "是",
            )
            for row in csv.DictReader(f)
        ]


def eval_question(groups: list[list[str]], ranked_paths: list[str],
                  ks: tuple[int, ...] = (3, 10)) -> dict:
    """单题指标:组间 AND(全中才 hit)、组内 OR(任一原子最早命中定 rank);
    MRR = 命中组 1/r 的组间平均(未命中组按 0 计入母),窗口 max(ks);D 桶空组 → {}。"""
    if not groups:
        return {}
    ranks = []
    for atoms in groups:
        r = next((i + 1 for i, p in enumerate(ranked_paths) if any(a in p for a in atoms)), None)
        ranks.append(r)
    out: dict[str, float] = {}
    kmax = max(ks)
    for k in ks:
        within = [r for r in ranks if r is not None and r <= k]
        out[f"hit@{k}"] = 1.0 if len(within) == len(ranks) else 0.0
        out[f"recall@{k}"] = len(within) / len(ranks)
    inv = [1.0 / r for r in ranks if r is not None and r <= kmax]
    out[f"mrr@{kmax}"] = sum(inv) / len(ranks)
    return out
