"""三道闸纯逻辑(spec §6)。embedding/search 由调用方注入,本文件零 I/O 零依赖。"""

from __future__ import annotations

import math
import re
import unicodedata

_PUNCT = re.compile(r"[\s，,。.、！!？?；;：:~～'\"“”‘’()（）【】\[\]《》<>—\-_+/|]+")


def normalize_question(q: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFKC", q))


def group_exact(rows: list[tuple[int, str]]) -> list[list[int]]:
    """闸1:归一化同问法并组,组内与组间都按行 id 升序(首现优先)。"""
    order: dict[str, list[int]] = {}
    for rid, q in sorted(rows):
        order.setdefault(normalize_question(q), []).append(rid)
    return list(order.values())


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def merge_semantic(keys: list[int], rep_vecs: dict[int, list[float]], thr: float) -> list[list[int]]:
    """闸2:单链接聚类(demo 规模 O(n²) 足够;宁严勿宽:归并即共享同一入池判定)。"""
    clusters: list[list[int]] = [[k] for k in keys]
    changed = True
    while changed:
        changed = False
        for x in range(len(clusters)):
            if changed:
                break
            for y in range(x + 1, len(clusters)):
                if any(cosine(rep_vecs[a], rep_vecs[b]) >= thr for a in clusters[x] for b in clusters[y]):
                    clusters[x] += clusters[y]
                    del clusters[y]
                    changed = True
                    break
    return clusters


def gate3_split(clusters: list[list[int]], rep_vecs: dict[int, list[float]],
                search_fn, thr: float) -> tuple[list[list[int]], list[list[int]]]:
    """闸3:簇代表向量对库 search,top1 ≥ thr → 整簇 discarded。search_fn(vec, k) -> [(id, score)]。"""
    kept: list[list[int]] = []
    dead: list[list[int]] = []
    for cluster in clusters:
        hits = search_fn(rep_vecs[cluster[0]], 1)
        (dead if hits and hits[0][1] >= thr else kept).append(cluster)
    return kept, dead
