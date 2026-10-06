"""ch10 T4:术语表驱动造数——真实池缺口由大模型照 17 类权威表补造。

prompt 携带术语表全文(类目+边界)+「两个及以上诉求」directive(验收 3 的语料来源);
断点续造:已有 synth.jsonl 按类计数,只补缺口。LLM 走项目 DashScope 通道
(app.services.chat_service.get_model),不占本对话账户。

实跑:uv run python -m finetune.synth --out finetune/drafts/synth.jsonl
控制台 ASCII;产物 UTF-8 JSONL。
"""

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.glossary import CLASSES, glossary_prompt_block, load_glossary

_PREFIX = re.compile(r"^\s*(?:\d+[.、．)）]?|[-*·])\s*")
MIN_LEN = 4


def build_prompt(cls_name: str, n: int) -> str:
    g = load_glossary()[cls_name]
    return (
        f"你是猫爪商城客服知识库的出题人。请生成 {n} 条顾客口吻的中文咨询问题,"
        f"本轮目标类目:「{cls_name}」——{g['boundary']}。\n"
        f"全量类目词表与边界(供多诉求句搭配参考):\n{glossary_prompt_block()}\n"
        "要求:1) 每行一条,不要编号、不要引号;2) 句式多样(询问/抱怨/操作请求),"
        "口语化、允许出现错别字;3) 约四分之一的句子要字面包含两个及以上诉求"
        "(如「买大了想退」=尺码+退换货),这些句必须同时明显命中「"
        f"{cls_name}」和词表里另一类;4) 不得出现真实姓名/电话/地址;"
        "5) 只输出问题本身,不要任何解释。"
    )


def parse_questions(reply: str) -> list[str]:
    out: list[str] = []
    for line in reply.splitlines():
        s = _PREFIX.sub("", line.strip()).strip()
        if len(s) >= MIN_LEN and not s.startswith(("```", "#")) and s not in out:
            out.append(s)
    return out


def count_synth(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    c: Counter[str] = Counter()
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            c[json.loads(ln)["main_class"]] += 1
    return dict(c)


def gap_plan(counts: dict[str, int], targets: dict[str, int]) -> dict[str, int]:
    return {c: targets[c] - counts.get(c, 0)
            for c in targets if targets[c] - counts.get(c, 0) > 0}


def default_targets(per_class: int) -> dict[str, int]:
    return {c: (max(40, per_class // 2) if c == "其他" else per_class)
            for c in CLASSES}


async def _run(args) -> int:
    model = get_frozen_model()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plan = gap_plan(count_synth(out), default_targets(args.target))
    if args.dry_run:
        print("[synth] dry-run plan:", json.dumps(plan, ensure_ascii=True))
        return 0
    for cls, need in sorted(plan.items()):
        got = 0
        attempt = 0
        while got < need and attempt < 12:
            attempt += 1
            chunk = min(25, need - got)  # 长生成拆小片,单次回复更稳
            try:
                r = await asyncio.wait_for(
                    model.ainvoke(build_prompt(cls, chunk)), timeout=240)
            except Exception:  # noqa: BLE001 —— 网络/超时均记attempt重试
                print(f"[synth] {cls.encode('ascii', 'replace').decode()} "
                      f"call-fail attempt={attempt}", flush=True)
                continue
            qs = parse_questions(r.content)
            with out.open("a", encoding="utf-8") as f:
                for q in qs[: need - got]:
                    f.write(json.dumps({"text": q, "src": "synth",
                                        "main_class": cls}, ensure_ascii=False) + "\n")
                    got += 1
            print(f"[synth] {cls.encode('ascii', 'replace').decode()} "
                  f"attempt={attempt} got={got}/{need}", flush=True)
        if got < need:
            print(f"[synth] WARN deficit {cls.encode('ascii', 'replace').decode()} "
                  f"{got}/{need}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="synth")
    p.add_argument("--target", type=int, default=100)
    p.add_argument("--out", default="finetune/drafts/synth.jsonl")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())


def get_frozen_model():
    """造数/预标专用通道:温度钉 0(终审 I-5——重跑标签必须逐字稳定)。"""
    from app.core.config import get_settings
    from app.services.chat_service import get_model

    st = get_settings().model_copy(update={"temperature": 0.0})
    return get_model(st)
