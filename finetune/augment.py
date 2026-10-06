"""ch10 T6 增强:同义词替换(×1)+句式前缀模板(半数 ×0.5),只扩 train。

类内同义词替换与中性前缀都不改诉求构成 → 标签守恒;valid/test 永不经此处。
"""

from finetune.glossary import CLASSES

_PREFIXES = ["你好,请问一下", "帮我看看:", "急,在线等:", "想咨询下,", "麻烦问下:"]


def _syn_swap(text: str, labels: list[str], glossary: dict) -> str | None:
    for lb in labels:
        syns = glossary[lb]["synonyms"]
        for syn in syns:
            if syn in text:
                repl = next((s for s in syns if s != syn), None)
                if repl:
                    return text.replace(syn, repl, 1)
    return None


def augment_train(train: list[dict], glossary: dict) -> list[dict]:
    aug: list[dict] = []
    for idx, r in enumerate(train):
        s = _syn_swap(r["text"], r["labels"], glossary)
        if s and s != r["text"]:
            aug.append({**r, "text": s, "aug": True})
        if idx % 2 == 0:
            pre = _PREFIXES[idx % len(_PREFIXES)]
            aug.append({**r, "text": pre + r["text"], "aug": True})
    return aug
