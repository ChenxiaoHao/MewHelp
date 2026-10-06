"""ch10 T5:LLM 预标(照术语表打标+顺手修错别字,拍板 2)+抽审面导出。

规则(需求书原文):字面提到几个诉求就打几个标签,一个不多一个不少;无法落类归「其他」。
实跑:uv run python -m finetune.prelabel  → finetune/drafts/labeled.jsonl
(真池 138 全量 + synth 草稿;并发≤4;断点续跑:已标 text 跳过)。控制台 ASCII。
"""

import argparse
import asyncio
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.glossary import CLASSES, glossary_prompt_block

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

PRELABEL_SYSTEM = (
    "你是电商客服问题标注员。对给定的一句顾客问题,依据 17 类权威类目表打标,"
    "并顺手修正句中错别字(不改语义、不改口吻)。\n类目表(名:边界):\n"
    + glossary_prompt_block()
    + "\n打标规则:字面提到几个诉求就打几个标签,一个不多一个不少;"
    "近邻边界按表内说明执行;确实无法落前 16 类的只打「其他」。"
    "只输出一个 JSON,不要任何解释:"
    '{"labels": ["类目名", ...], "text": "修正错别字后的句子"}'
)


def parse_prelabel(reply: str) -> dict:
    m = _FENCE.search(reply)
    body = m.group(1) if m else reply
    try:
        d = json.loads(body.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"预标回复非 JSON: {reply[:60]!r}") from exc
    raw = d.get("labels")
    if not isinstance(raw, list):
        raise ValueError("labels 字段缺失或非列表")
    labels = [x for x in raw if x in CLASSES]
    if not labels:
        labels = ["其他"]
    out = {"labels": list(dict.fromkeys(labels))}
    t = d.get("text")
    if isinstance(t, str) and t.strip():
        out["text"] = t.strip()
    return out


def parse_labels(reply: str) -> list[str]:
    return parse_prelabel(reply)["labels"]


def run_audit_export(rows: list[dict], out_path: Path,
                     per_class: float = 0.1, seed: int = 7) -> Path:
    """抽审面=每类单标签样本 10% + 全部多标签样本;同 seed 确定性。"""
    rng = random.Random(seed)
    picked: list[dict] = []
    by_class: dict[str, list[dict]] = {c: [] for c in CLASSES}
    for r in rows:
        if len(r["labels"]) > 1:
            picked.append(r)
            continue
        by_class[r["labels"][0]].append(r)
    for c, bucket in by_class.items():
        single = [r for r in bucket if r not in picked]
        k = max(1, round(len(single) * per_class)) if single else 0
        picked.extend(sorted(rng.sample(single, k), key=lambda r: r["text"]))
    picked.sort(key=lambda r: (r["labels"][0], r["text"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["text", "labels", "src", "reviewed_labels"])
        for r in picked:
            w.writerow([r["text"], "|".join(r["labels"]), r.get("src", ""), ""])
    return out_path


async def _load_pool_texts() -> list[dict]:
    from sqlalchemy import select

    from app.core.config import get_settings
    from app.db.engine import dispose_engine, get_session_factory, init_engine
    from app.db.models import LowConfidenceQuestion

    from finetune.clean import clean

    init_engine(get_settings())
    try:
        async with get_session_factory()() as s:
            rows = (await s.execute(select(LowConfidenceQuestion.raw_question
                                            ).order_by(LowConfidenceQuestion.id))
                    ).scalars().all()
    finally:
        await dispose_engine()
    return [{"text": clean(t), "src": "pool"} for t in rows]


async def _run(args) -> int:
    from app.core.config import get_settings
    from app.services.chat_service import get_model

    from finetune.synth import parse_questions  # noqa: F401  仅同源语料面

    model = get_model(get_settings())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(ln)["text"] for ln in
                out.read_text(encoding="utf-8").splitlines() if ln.strip()}
    corpus = await _load_pool_texts()
    if Path(args.synth).exists():
        corpus += [{"text": json.loads(ln)["text"], "src": "synth"}
                   for ln in Path(args.synth).read_text(encoding="utf-8").splitlines()
                   if ln.strip()]
    sem = asyncio.Semaphore(4)

    async def one(item: dict) -> None:
        if item["text"] in done:
            return
        async with sem:
            for _ in range(3):
                try:
                    r = await model.ainvoke(
                        [{"role": "system", "content": PRELABEL_SYSTEM},
                         {"role": "user", "content": item["text"]}])
                    d = parse_prelabel(r.content)
                    break
                except ValueError:
                    await asyncio.sleep(1)
            else:
                print(f"[prelabel] SKIP unparseable (id={hash(item['text']) % 10**6})",
                      flush=True)
                return
        rec = {"text": d.get("text", item["text"]), "labels": d["labels"],
               "src": item["src"]}
        if d.get("text"):
            rec["typo_fixed"] = d["text"] != item["text"]
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(it) for it in corpus))
    total = sum(1 for _ in out.open(encoding="utf-8"))
    print(f"[prelabel] labeled total={total} of corpus={len(corpus)}", flush=True)
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    p = run_audit_export(rows, Path("finetune/audit/annotation_review.csv"),
                         per_class=args.audit_ratio, seed=args.seed)
    print(f"[prelabel] audit csv -> {p.as_posix()}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="prelabel")
    p.add_argument("--out", default="finetune/drafts/labeled.jsonl")
    p.add_argument("--synth", default="finetune/drafts/synth.jsonl")
    p.add_argument("--audit-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    return asyncio.run(_run(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
