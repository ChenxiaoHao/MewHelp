"""ch10 T8 评测:封存 test 一次性跑——逐类 P/R/F1 + 每类二值混淆矩阵 + 错例台账。

用法:uv run python -m finetune.evaluate [--model models/ch10_topic] [--data finetune/dataset/test.jsonl]
阈值只读 topic_config.json(与推理/训练同源,禁硬编码)。
控制台一行 ASCII;中文报告 UTF-8 落 finetune/reports/ch10_eval_report.md,
错例全量落 misclassified.csv(utf-8-sig 供人工抽判)。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.glossary import CLASSES  # noqa: E402

REPORT = Path("finetune/reports/ch10_eval_report.md")
MISCSV = Path("finetune/reports/misclassified.csv")


def to_multi_hot(labels: list[str], classes: list[str]) -> list[int]:
    return [1 if c in labels else 0 for c in classes]


def binary_confusion(preds: list[list[int]], golds: list[list[int]],
                     classes: list[str]) -> dict[str, dict]:
    P = np.asarray(preds, dtype=np.int8)
    G = np.asarray(golds, dtype=np.int8)
    out: dict[str, dict] = {}
    for i, c in enumerate(classes):
        p, g = P[:, i], G[:, i]
        out[c] = {"tp": int(((p == 1) & (g == 1)).sum()),
                  "fp": int(((p == 1) & (g == 0)).sum()),
                  "fn": int(((p == 0) & (g == 1)).sum()),
                  "tn": int(((p == 0) & (g == 0)).sum())}
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def micro_f1(conf: dict[str, dict]) -> float:
    tp = sum(c["tp"] for c in conf.values())
    fp = sum(c["fp"] for c in conf.values())
    fn = sum(c["fn"] for c in conf.values())
    return prf(tp, fp, fn)[2]


def misclassified_rows(rows: list[dict], preds: list[list[int]]) -> list[dict]:
    out = []
    for r, pr in zip(rows, preds):
        gold = to_multi_hot(r["labels"], CLASSES)
        if pr != gold:
            out.append({"text": r["text"],
                        "gold": "|".join(r["labels"]),
                        "pred": "|".join(c for c, v in zip(CLASSES, pr) if v)
                        or "其他"})  # 与部署端零命中兜底同口径
    return out


def report_text(conf: dict, threshold: float, n: int,
                multi_hits: list[dict], trained_at: str = "") -> str:
    lines = ["# ch10 主题分类器评测报告(封存 test 集)", "",
             f"- 样本数:{n} · 阈值:{threshold:.2f}(valid 扫描,config 同源)",
             f"- micro-F1:{micro_f1(conf):.4f} · macro-F1:"
             f"{np.mean([prf(**{k: conf[c][k] for k in ('tp', 'fp', 'fn')}) for c in CLASSES]).item():.4f}",
             "", "| 类目 | TP | FP | FN | TN | P | R | F1 | support |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in CLASSES:
        d = conf[c]
        p, r, f = prf(d["tp"], d["fp"], d["fn"])
        lines.append(f"| {c} | {d['tp']} | {d['fp']} | {d['fn']} | {d['tn']} | "
                     f"{p:.3f} | {r:.3f} | {f:.3f} | {d['tp'] + d['fn']} |")
    lines += ["", "## 多标签命中例(验收 3)", ""]
    for m in multi_hits[:5]:
        lines.append(f"- 「{m['text']}」→ 预测 {m['pred']}(金标 {m['gold']})")
    lines += ["", f"- 模型训练于:{trained_at}" if trained_at else "", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evaluate")
    ap.add_argument("--model", type=Path, default=Path("models/ch10_topic"))
    ap.add_argument("--data", type=Path, default=Path("finetune/dataset/test.jsonl"))
    args = ap.parse_args(argv)

    cfg = json.loads((args.model / "topic_config.json").read_text(encoding="utf-8"))
    assert cfg["classes"] == CLASSES, "词表漂移红线:config 与 glossary 必须同序同名"
    threshold = float(cfg["threshold"])

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model))
    model.eval()
    rows = [json.loads(ln) for ln in
            args.data.read_text(encoding="utf-8").splitlines() if ln.strip()]
    preds: list[list[int]] = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            batch = [r["text"] for r in rows[i:i + 32]]
            enc = tok(batch, truncation=True, max_length=128, padding=True,
                      return_tensors="pt")
            probs = torch.sigmoid(model(**enc).logits).numpy()
            preds += [[int(v) for v in (p >= threshold)] for p in probs]
    golds = [to_multi_hot(r["labels"], CLASSES) for r in rows]
    conf = binary_confusion(preds, golds, CLASSES)
    multi_hits = [{"text": r["text"], "gold": "|".join(r["labels"]),
                   "pred": "|".join(c for c, v in zip(CLASSES, pr) if v)}
                  for r, pr in zip(rows, preds)
                  if sum(pr) >= 2 and sum(to_multi_hot(r["labels"], CLASSES)) >= 2]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text(conf, threshold, len(rows), multi_hits,
                                  trained_at=str(cfg.get("trained_at", ""))),
                      encoding="utf-8")
    with MISCSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["text", "gold", "pred"])
        w.writeheader()
        w.writerows(misclassified_rows(rows, preds))
    macro = float(np.mean([prf(conf[c]["tp"], conf[c]["fp"], conf[c]["fn"])[2]
                           for c in CLASSES]))
    print(f"[eval] n={len(rows)} threshold={threshold:.2f} macro_f1={macro:.4f} "
          f"micro_f1={micro_f1(conf):.4f} mis={len(misclassified_rows(rows, preds))} "
          f"report={REPORT.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
