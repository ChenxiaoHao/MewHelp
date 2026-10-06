"""ch10 T5:预标解析四类脏回复钉 + 抽审导出确定性 + apply_audit 回写钉(unit)。

实跑全量预标是「样例/评估集验证」步骤,不在单测(不烧 API 额度)。
"""

import json

import pytest

from finetune.glossary import CLASSES
from finetune.prelabel import PRELABEL_SYSTEM, parse_labels, parse_prelabel


def test_system_prompt_carries_glossary_and_typo_duty():
    assert "错别字" in PRELABEL_SYSTEM
    assert all(c in PRELABEL_SYSTEM for c in CLASSES)
    assert "labels" in PRELABEL_SYSTEM


def test_parse_labels_plain_and_fenced():
    assert parse_labels('{"labels": ["发票", "运费"]}') == ["发票", "运费"]
    assert parse_labels('```json\n{"labels": ["发票"]}\n```') == ["发票"]


def test_parse_labels_dedup_and_overclass_dropped():
    assert parse_labels('{"labels": ["物流", "物流", "配送"]}') == ["物流"]
    assert parse_labels('{"labels": ["越界词", "玄学"]}') == ["其他"]
    assert parse_labels('{"labels": []}') == ["其他"]


def test_parse_labels_broken_raises():
    with pytest.raises(ValueError):
        parse_labels("我直接给你几个词:发票、运费")


def test_parse_prelabel_returns_text_fix():
    d = parse_prelabel('{"labels": ["尺码"], "text": "买大了想退"}')
    assert d["labels"] == ["尺码"] and d["text"] == "买大了想退"
    assert parse_labels('{"labels": ["尺码"]}') == ["尺码"]  # 无 text 键容忍


ROWS = [{"text": f"句{i}", "labels": (["物流"] if i % 3 else ["物流", "运费"]), "src": "synth"}
        for i in range(30)]


def test_audit_export_deterministic_full_multilabel(tmp_path):
    from finetune.prelabel import run_audit_export
    p1 = run_audit_export(ROWS, tmp_path / "a1.csv", per_class=0.1, seed=7)
    p2 = run_audit_export(ROWS, tmp_path / "a2.csv", per_class=0.1, seed=7)
    assert p1.read_bytes() == p2.read_bytes(), "同 seed 必同抽审面"
    import csv
    rows = list(csv.DictReader(p1.open(encoding="utf-8-sig")))
    multi = {r["text"] for r in rows if "|" in r["labels"]}
    assert multi == {r["text"] for r in ROWS if len(r["labels"]) > 1}, "多标签 100% 入审"
    assert all("reviewed_labels" in r for r in rows)


def test_apply_audit_rejects_illegal_and_writes_back(tmp_path):
    from finetune.apply_audit import apply_audit
    ds = tmp_path / "labeled.jsonl"
    lines = [json.dumps(r, ensure_ascii=False) for r in ROWS]
    ds.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit = tmp_path / "aud.csv"
    import csv
    with audit.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["text", "labels", "src", "reviewed_labels"])
        w.writerow([ROWS[0]["text"], "物流", "synth", "发票|运费"])
        w.writerow([ROWS[1]["text"], "物流|运费", "synth", "玄学类"])
    with pytest.raises(ValueError):
        apply_audit(audit, ds)  # 越类名整体拒收,不落半改
    txt = ds.read_text(encoding="utf-8")
    assert '"labels": ["物流"]' in txt  # 未被部分写坏

    audit2 = tmp_path / "aud2.csv"
    with audit2.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["text", "labels", "src", "reviewed_labels"])
        w.writerow([ROWS[0]["text"], "物流", "synth", "发票|运费"])
        w.writerow([ROWS[2]["text"], "物流", "synth", "  "])  # 空=不改
    n = apply_audit(audit2, ds)
    assert n == 1
    got = [json.loads(x) for x in ds.read_text(encoding="utf-8").splitlines()]
    assert got[0]["labels"] == ["发票", "运费"]
    assert got[2]["labels"] == ["物流"]
