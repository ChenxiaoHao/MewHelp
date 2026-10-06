"""ch10 T7 smoke:integration——tiny 数据 --smoke --max-steps 2 跑通全链路,断言三件套+config。

跑法(模型资产就绪后):uv run pytest tests/test_train_smoke_ch10.py -m integration -v
"""

import json
from pathlib import Path

import pytest

from finetune.glossary import CLASSES
from finetune.train import main


def _write_rows(path: Path, n: int):
    import random

    rng = random.Random(7)
    lines = []
    for i in range(n):
        labels = [CLASSES[i % len(CLASSES)]]
        if i % 4 == 0:
            labels.append(CLASSES[(i + 3) % len(CLASSES)])
        lines.append(json.dumps(
            {"text": f"测试句子{i}关于{labels[0]}的问题呀", "labels": labels},
            ensure_ascii=False))
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.integration
def test_smoke_produces_model_assets(tmp_path):
    data, out = tmp_path / "dataset", tmp_path / "out"
    data.mkdir()
    _write_rows(data / "train.jsonl", 24)
    _write_rows(data / "valid.jsonl", 8)
    rc = main(["--smoke", "--max-steps", "2", "--data-dir", str(data),
               "--out-dir", str(out), "--runs-dir", str(tmp_path / "runs")])
    assert rc == 0
    assert (out / "config.json").exists()
    assert list(out.glob("*.safetensors")), "缺模型权重"
    assert (out / "tokenizer.json").exists()
    cfg = json.loads((out / "topic_config.json").read_text(encoding="utf-8"))
    assert cfg["classes"] == CLASSES
    assert 0.30 <= cfg["threshold"] <= 0.70
