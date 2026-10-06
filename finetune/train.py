"""ch10 T7 训练:RoBERTa-wwm-ext 全参微调(不用 LoRA)+早停+valid 扫阈值。

用法:uv run python -m finetune.train [--epochs 10] [--smoke] [--max-steps 2]
产物:models/ch10_topic/(模型+tokenizer)+ topic_config.json
      {classes(定序), threshold(valid macro-F1 最优,扫 [0.30,0.70] 步 0.05),
       best_valid_f1, trained_at}
HF 下载走 hf-mirror(拍板 6)。控制台 ASCII;训练日志 ASCII 落
finetune/reports/train_log.txt(运行 shell tee)。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,  # noqa: E402
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          TrainingArguments, Trainer)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.glossary import CLASSES  # noqa: E402

MODEL_SRC = "hfl/chinese-roberta-wwm-ext"
OUT_DIR = Path("models/ch10_topic")
RUNS_DIR = Path("models/.ch10_runs")
MAX_LEN = 128


def load_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in
            Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def make_label_vec(labels: list[str]) -> list[float]:
    vec = [0.0] * len(CLASSES)
    for lb in labels:
        vec[CLASSES.index(lb)] = 1.0
    return vec


class MultiLabelDS(Dataset):
    def __init__(self, rows: list[dict], tokenizer):
        self.rows = rows
        self.tok = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=MAX_LEN)
        enc = {k: torch.tensor(v) for k, v in enc.items()}
        enc["labels"] = torch.tensor(make_label_vec(r["labels"]))
        return enc


def _f1_from_counts(tp, fp, fn):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 2 * tp / (2 * tp + fp + fn)
    return np.where(np.isnan(out), 0.0, out)


def compute_metrics(eval_pred, threshold: float = 0.5) -> dict:
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.float32)
    tp = ((preds == 1) & (labels == 1)).sum(0)
    fp = ((preds == 1) & (labels == 0)).sum(0)
    fn = ((preds == 0) & (labels == 1)).sum(0)
    per = _f1_from_counts(tp, fp, fn)
    macro = float(per.mean())
    micro = float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()))
    return {"f1_macro": macro, "f1_micro": micro}


def sweep_threshold(model, tokenizer, rows: list[dict], device) -> tuple[float, float]:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            batch = [r["text"] for r in rows[i:i + 32]]
            enc = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            probs.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    P = np.vstack(probs)
    G = np.array([make_label_vec(r["labels"]) for r in rows])
    best_th, best_f1 = 0.5, -1.0
    for th in np.arange(0.30, 0.7001, 0.05):
        f1 = compute_metrics((P, G), threshold=float(th))["f1_macro"]
        if f1 > best_f1:
            best_th, best_f1 = float(th), f1
    return best_th, best_f1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="train")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--data-dir", default="finetune/dataset")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--runs-dir", default=str(RUNS_DIR))
    args = p.parse_args(argv)
    out_dir, runs_dir = Path(args.out_dir), Path(args.runs_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_SRC)
    id2label = {i: c for i, c in enumerate(CLASSES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_SRC, num_labels=len(CLASSES),
        problem_type="multi_label_classification",  # v5 枚举名(非 v4 脚本映射名)
        id2label=id2label, label2id={c: i for i, c in enumerate(CLASSES)})

    train_rows = load_rows(Path(args.data_dir) / "train.jsonl")
    valid_rows = load_rows(Path(args.data_dir) / "valid.jsonl")
    if args.smoke:
        train_rows, valid_rows = train_rows[:24], valid_rows[:8]
        epochs, max_steps = 1, args.max_steps if args.max_steps > 0 else 2
        batch = 4
    else:
        epochs, max_steps, batch = args.epochs, args.max_steps, args.batch

    targs = TrainingArguments(
        output_dir=str(runs_dir),
        learning_rate=args.lr, per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch * 2,
        num_train_epochs=epochs, weight_decay=0.01, max_steps=max_steps,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro",
        greater_is_better=True, save_total_limit=2,
        fp16=(device == "cuda"), logging_steps=10,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=MultiLabelDS(train_rows, tok),
        eval_dataset=MultiLabelDS(valid_rows, tok),
        processing_class=tok, data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    ev = trainer.evaluate()
    best_f1 = ev.get("eval_f1_macro", 0.0)

    th, sweep_f1 = sweep_threshold(trainer.model, tok, valid_rows, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "topic_config.json").write_text(json.dumps({
        "classes": CLASSES, "threshold": th,
        "best_valid_f1": max(float(best_f1), sweep_f1),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] device={device} eval_f1_macro={float(best_f1):.4f} "
          f"sweep_f1={sweep_f1:.4f} threshold={th:.2f} out={out_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
