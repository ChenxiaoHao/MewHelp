"""ch09 T4:evidence_confidence 纯函数闸核(spec 置信度闸升级节)。

从 state["evidence"](dict 或 ScoredRow 兼容)算三信号——top1 /
n_eff(score ≥ floor_eff 条数)/ gap(top1−top2,单证据 gap=top1)。
判定形态 rule|sum 与全部参数由 300 题校准脚本(evals/calibrate_confidence.py)
定形定值,落 config `evidence_conf_*` 键组;缺键夹具回落 rule 形单阈值
(retrieval_low_conf_threshold,默认 0.161)= ch05 老行为零漂移。
位置与行为不变:仍卡 retrieve 后、进 agent 前;空证据必拦;RRF 降级带
旁路语义在 nodes 闸节点保留(本模块只出判定,不碰旁路)。
"""

from __future__ import annotations

from dataclasses import dataclass

LEGACY_THRESHOLD = 0.161  # ch04 闸1 终值(缺键兜底,与 Settings 默认一致)

# sum 形归一根(与校准脚本共用口径:计数/间隙先化到 [0,1])
N_CAP = 3
GAP_CAP = 0.5


@dataclass(frozen=True)
class EvidenceVerdict:
    ok: bool
    conf: float
    detail: str


def _scores(items) -> list[float]:
    out = []
    for i in items or []:
        s = i["score"] if isinstance(i, dict) else i.score
        out.append(float(s))
    return out


def signals_from_evidence(evidence, floor_eff: float) -> tuple[float, int, float]:
    """(top1, n_eff, gap);空证据 → (0.0, 0, 0.0)。"""
    scores = sorted(_scores(evidence), reverse=True)
    if not scores:
        return 0.0, 0, 0.0
    top1 = scores[0]
    gap = top1 - scores[1] if len(scores) > 1 else top1
    n_eff = sum(1 for s in scores if s >= floor_eff)
    return top1, n_eff, gap


def _k(cfg, name: str, fallback):
    v = getattr(cfg, f"evidence_conf_{name}", None)
    return fallback if v is None else v


def evaluate(evidence, cfg) -> EvidenceVerdict:
    """f(top1,n_eff,gap) → 过闸判定;cfg=Settings(或任意含键对象)。

    rule: top1≥θ 且 n_eff≥n_min 且 gap≥gap_min(conf 报 top1,复盘主量);
    sum : conf=w1·top1+wn·min(n,N_CAP)/N_CAP+wg·min(gap,GAP_CAP)/GAP_CAP≥θ。
    空证据两形皆拦(spec:与 retriever 闸1 交互不变)。
    """
    legacy = getattr(cfg, "retrieval_low_conf_threshold", None)
    if legacy is None:
        legacy = LEGACY_THRESHOLD
    theta = float(_k(cfg, "threshold", legacy))
    floor = float(_k(cfg, "floor_eff", theta))
    form = str(_k(cfg, "form", "rule"))
    top1, n, gap = signals_from_evidence(evidence, floor)
    sig = f"(top1={top1:.4f},n={n},gap={gap:.4f})"
    if form == "sum":
        w1 = float(_k(cfg, "w_top1", 1.0))
        wn = float(_k(cfg, "w_n", 0.0))
        wg = float(_k(cfg, "w_gap", 0.0))
        conf = w1 * top1 + wn * min(n, N_CAP) / N_CAP + wg * min(gap, GAP_CAP) / GAP_CAP
        ok = conf >= theta
        cmp_ = ">=" if ok else "<"
        detail = f"conf={conf:.4f}{cmp_}{theta:.4f}{sig}"
    else:
        n_min = int(_k(cfg, "n_eff_min", 1))
        gap_min = float(_k(cfg, "gap_min", 0.0))
        conf = top1
        ok = top1 >= theta and n >= n_min and gap >= gap_min
        if ok:
            detail = f"conf={conf:.4f}>={theta:.4f}{sig}"
        else:
            why = []
            if top1 < theta:
                why.append(f"top1<{theta:.4f}")
            if n < n_min:
                why.append(f"n<{n_min}")
            if gap < gap_min:
                why.append(f"gap<{gap_min:.4f}")
            detail = f"conf={conf:.4f}<{theta:.4f}{sig}{'[' + '+'.join(why) + ']' if why else ''}"
    return EvidenceVerdict(ok=ok, conf=round(conf, 4), detail=detail)
