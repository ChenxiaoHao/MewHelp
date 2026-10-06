"""17 类权威术语表:标注/造数/增强/id2label 四处共用的唯一词表。

名单与顺序以本目录 glossary.json 为准(spec「类目契约」节);漂移由
tests/test_glossary_ch10.py 钉死。
"""

import json
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "glossary.json"
_cache: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _cache
    if _cache is None:
        raw = json.loads(_PATH.read_text(encoding="utf-8"))
        _cache = {c["name"]: {"boundary": c["boundary"], "synonyms": list(c["synonyms"])}
                  for c in raw["classes"]}
    return _cache


CLASSES: list[str] = [c["name"] for c in
                      json.loads(_PATH.read_text(encoding="utf-8"))["classes"]]


def load_glossary() -> dict[str, dict]:
    return _load()


def assert_valid_labels(labels: list[str]) -> list[str]:
    """白名单校验+按输入序去重;越类名/空集即炸(预标解析唯一入口)。"""
    seen: list[str] = []
    for lb in labels:
        if lb not in CLASSES:
            raise ValueError(f"越类标签: {lb!r} 不在 17 类权威表")
        if lb not in seen:
            seen.append(lb)
    if not seen:
        raise ValueError("labels 不得为空;无诉求句应归「其他」")
    return seen


def glossary_prompt_block() -> str:
    """术语表全文(类目+边界说明),造数/预标 prompt 共用。"""
    g = _load()
    lines = [f"- {name}:{g[name]['boundary']}" for name in CLASSES]
    return "\n".join(lines)
