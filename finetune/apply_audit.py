"""ch10 T5 回改面:人工抽审 CSV 的 reviewed_labels 回写 labeled.jsonl。

整批原子:任一行非法类名即全批拒收(不落半改);reviewed 空=不改。
用法:uv run python -m finetune.apply_audit --audit finetune/audit/annotation_review.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.glossary import CLASSES


def apply_audit(audit_csv: Path, dataset: Path) -> int:
    edits: dict[str, list[str]] = {}
    with audit_csv.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rev = (r.get("reviewed_labels") or "").strip()
            if not rev:
                continue
            labels = [x.strip() for x in rev.split("|") if x.strip()]
            bad = [x for x in labels if x not in CLASSES]
            if bad or not labels:
                raise ValueError(
                    f"非法抽审标签 {bad or '(空)'} 于行 {r.get('text', '')!r}")
            edits[r["text"]] = list(dict.fromkeys(labels))
    if not edits:
        return 0
    rows = [json.loads(ln)
            for ln in dataset.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n = 0
    for row in rows:
        if row["text"] in edits:
            row["labels"] = edits[row["text"]]
            n += 1
    dataset.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                       encoding="utf-8")
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="apply_audit")
    p.add_argument("--audit", type=Path,
                   default=Path("finetune/audit/annotation_review.csv"))
    p.add_argument("--dataset", type=Path,
                   default=Path("finetune/drafts/labeled.jsonl"))
    args = p.parse_args(argv)
    n = apply_audit(args.audit, args.dataset)
    print(f"[apply_audit] rows_updated={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
