"""ch10 T6 切分:按 label-组合分层 80/10/10;稀有组合(<3 行)轮转分配。

test 集切完即封存:训练/调阈值/增强一律不得触碰(泄漏红线)。
CLI(验证步骤,labeled 就绪后):
  uv run python -m finetune.split --in finetune/drafts/labeled.jsonl --out-dir finetune/dataset
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def stratified_split(rows: list[dict], ratios: tuple = (0.8, 0.1, 0.1),
                     seed: int = 7) -> dict[str, list]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(sorted(r["labels"]))].append(dict(r))
    rng = random.Random(seed)
    out: dict[str, list] = {"train": [], "valid": [], "test": []}
    names = ("train", "valid", "test")
    rr = 0
    for key in sorted(groups):
        g = groups[key]
        rng.shuffle(g)
        if len(g) < 3:  # 孤例组合桶:轮转铺开,保证稀有类不整组沉入单一集
            for r in g:
                out[names[rr % 3]].append(r)
                rr += 1
            continue
        n_te = max(1, round(len(g) * ratios[2]))
        n_va = max(1, round(len(g) * ratios[1]))
        out["test"] += g[:n_te]
        out["valid"] += g[n_te:n_te + n_va]
        out["train"] += g[n_te + n_va:]
    for k in names:
        out[k].sort(key=lambda r: (r["labels"][0], r["text"]))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="split")
    p.add_argument("--in", dest="src", type=Path,
                   default=Path("finetune/drafts/labeled.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("finetune/dataset"))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    rows = [json.loads(ln) for ln in
            args.src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = stratified_split(rows, seed=args.seed)
    from finetune.augment import augment_train
    from finetune.glossary import load_glossary
    out["train"] += augment_train(out["train"], load_glossary())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in out.items():
        (args.out_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part),
            encoding="utf-8")
    for c in sorted({lb for r in rows for lb in r["labels"]}):
        counts = " ".join(str(sum(1 for r in part if c in r["labels"]))
                          for part in (out["train"], out["valid"], out["test"]))
        print(f"[split] cls#{hash(c) % 1000:03d} train/valid/test={counts}")
    print(f"[split] total in={len(rows)} "
          f"train={len(out['train'])} valid={len(out['valid'])} test={len(out['test'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
