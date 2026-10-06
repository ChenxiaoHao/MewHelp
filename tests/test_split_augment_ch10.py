"""ch10 T6:分层切分比例/防泄漏 + 增强只扩 train/标签守恒(unit)。

实跑出定稿 dataset 是验证步骤(labeled.jsonl 就绪后)。
"""

import json
from collections import Counter

import pytest

from finetune.augment import augment_train
from finetune.glossary import CLASSES, load_glossary
from finetune.split import stratified_split


def _rows(n_per_class=40):
    g = load_glossary()
    rows = []
    for ci, c in enumerate(CLASSES):
        word = g[c]["synonyms"][0]
        for i in range(n_per_class):
            if i % 4 == 0 and ci < 4:  # 前四大类掺多标签组合(物流+运费双词面)
                rows.append({"text": f"句{ci}_{i}_{g['物流']['synonyms'][0]}"
                                     f"{g['运费']['synonyms'][0]}",
                             "labels": ["物流", "运费"], "src": "synth"})
            else:
                rows.append({"text": f"句{ci}_{i}_问下{word}", "labels": [c],
                             "src": "synth"})
    return rows


def test_split_ratios_and_disjoint_no_leak():
    out = stratified_split(_rows(), seed=7)
    train, valid, test = (out["train"], out["valid"], out["test"])
    assert len(train) + len(valid) + len(test) == sum(len(v) for v in out.values())
    ts, vs, xs = (Counter(r["text"] for r in train), Counter(r["text"] for r in valid),
                  Counter(r["text"] for r in test))
    assert not (set(ts) & set(vs)) and not (set(ts) & set(xs)) and not (set(vs) & set(xs)), \
        "一行不得跨集重复"
    total_all = Counter()
    for r in _rows():
        total_all[r["labels"][0]] += 1
    for c in CLASSES:
        tc = sum(1 for r in train if c in r["labels"])
        vc = sum(1 for r in valid if c in r["labels"])
        xc = sum(1 for r in test if c in r["labels"])
        n = total_all[c]
        assert tc >= n * 0.70 and vc >= n * 0.05 and xc >= n * 0.05, c


def test_split_deterministic_same_seed():
    a = stratified_split(_rows(), seed=42)
    b = stratified_split(_rows(), seed=42)
    assert [r["text"] for r in a["train"]] == [r["text"] for r in b["train"]]


def test_singleton_combo_bucket_spread():
    rows = _rows(n_per_class=12)
    odd = [{"text": "孤例组合句一", "labels": ["保修维修", "价保"], "src": "synth"}]
    out = stratified_split(rows + odd, seed=3)
    assert any(r["text"] in {o["text"] for o in odd}
               for r in out["valid"] + out["test"] + out["train"])


def test_augment_only_train_labels_preserved_no_collision():
    train, _, test = (lambda o: (o["train"], o["valid"], o["test"]))(
        stratified_split(_rows(), seed=7))
    aug = augment_train(train, load_glossary())
    n = len(train)
    assert 1.2 * n <= len(aug) <= 2.2 * n, f"增强量 {len(aug)}/{n} 越界"
    assert all(r.get("aug") is True for r in aug)
    orig_texts = {r["text"] for r in train}
    test_texts = {r["text"] for r in test}
    for r in aug:
        assert r["text"] not in orig_texts and r["text"] not in test_texts
        assert all(lb in CLASSES for lb in r["labels"])
        src = next((t for t in train if t["labels"] == r["labels"]), None)
        assert src is not None, "标签必须守恒于某原行"
