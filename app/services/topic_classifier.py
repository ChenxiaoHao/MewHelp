"""ch10 T9 旁路推理:进程内一次性加载 models/ch10_topic,批量归 17 类主题。

唯一使用方是 app/jobs/topic_classify.py——实时主链路零 import 零调用。
阈值/类目序只认 topic_config.json(与训练/评测同源)。
"""

from __future__ import annotations

import json
from pathlib import Path

from finetune.glossary import CLASSES


def labels_from_probs(probs: list[float], classes: list[str],
                      threshold: float) -> list[str]:
    """sigmoid 概率 → 标签集(类序保序);零命中兜底「其他」。"""
    hit = [c for c, p in zip(classes, probs) if p >= threshold]
    return hit or ["其他"]


class TopicClassifier:
    def __init__(self, model_dir: str | Path = "models/ch10_topic",
                 device: str | None = None):
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)

        p = Path(model_dir)
        cfg = json.loads((p / "topic_config.json").read_text(encoding="utf-8"))
        if cfg["classes"] != CLASSES:
            raise RuntimeError("词表漂移红线:topic_config.classes != glossary.CLASSES")
        self.classes: list[str] = cfg["classes"]
        self.threshold: float = float(cfg["threshold"])
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(str(p))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(p))
        self.model.to(self.device).eval()

    def classify_batch(self, texts: list[str], batch_size: int = 32) -> list[list[str]]:
        out: list[list[str]] = []
        torch = self._torch
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], truncation=True,
                               max_length=128, padding=True,
                               return_tensors="pt").to(self.device)
                probs = torch.sigmoid(self.model(**enc).logits).cpu().tolist()
                out += [labels_from_probs(pr, self.classes, self.threshold)
                        for pr in probs]
        return out
