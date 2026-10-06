"""ch10 T8:评测纯函数钉——二值混淆四格已知值/prf 零分母/report 成形/错例判定。

真实 test 首跑是验证步骤(模型资产就绪后),不在单测。
"""

from finetune.evaluate import (binary_confusion, micro_f1, misclassified_rows,
                               prf, report_text, to_multi_hot)
from finetune.glossary import CLASSES

C = ["物流", "运费", "发票"]


def _mk_rows():
    return [{"text": "快递和邮费都要问", "labels": ["物流", "运费"]},
            {"text": "发票咋开", "labels": ["发票"]},
            {"text": "快递到哪了", "labels": ["物流"]},
            {"text": "运费谁出", "labels": ["发票"]}]  # 金标故意刁钻:文本运费,标发票


def _preds():
    return [["物流", "运费"], ["发票"], ["发票"], ["发票"]]


def test_to_multi_hot_order():
    vec = to_multi_hot(["运费", "发票"], CLASSES)
    assert sum(vec) == 2 and vec[CLASSES.index("运费")] == 1


def test_binary_confusion_known_values():
    conf = binary_confusion([to_multi_hot(p, C) for p in _preds()],
                            [to_multi_hot(r["labels"], C) for r in _mk_rows()], C)
    assert conf["物流"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 2}
    assert conf["运费"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 3}
    assert conf["发票"] == {"tp": 2, "fp": 1, "fn": 0, "tn": 1}


def test_prf_zero_denominator_safe():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    p, r, f = prf(1, 1, 1)
    assert abs(p - 0.5) < 1e-9 and abs(r - 0.5) < 1e-9 and abs(f - 0.5) < 1e-9


def test_micro_f1_and_report_shape():
    conf = binary_confusion([to_multi_hot(p, C) for p in _preds()],
                            [to_multi_hot(r["labels"], C) for r in _mk_rows()], C)
    assert 0.0 <= micro_f1(conf) <= 1.0
    full = {c: conf.get(c, {"tp": 0, "fp": 0, "fn": 0, "tn": 0}) for c in CLASSES}
    md = report_text(full, 0.45, 4, [{"text": "买大了想退", "gold": "尺码|退换货",
                                      "pred": "尺码|退换货"}])
    assert "| 物流 |" in md and "micro-F1" in md and "验收 3" in md
    assert "买大了想退" in md


def test_misclassified_rows_detects_diff():
    rows = _mk_rows()
    preds = [to_multi_hot(p, CLASSES) for p in _preds()]
    mis = misclassified_rows(rows, preds)
    assert len(mis) == 1 and mis[0]["text"] == "快递到哪了"
    assert mis[0]["gold"] == "物流" and mis[0]["pred"] == "发票"
    # 全对则空
    perfect = [to_multi_hot(r["labels"], CLASSES) for r in rows]
    assert misclassified_rows(rows, perfect) == []
