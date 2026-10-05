"""ch09 T3 接缝:09/10 DDL ⊇ ORM(Review Focus 2 的纸面半)。

down -v 全新重建只按 db/init 升序执行;09/10 若缺位或列名/ENUM 值漂移,
池写/快照写会「模型有列、库里没列」或中文 ENUM 双编码乱值。三面并钉:
①新表列名集 ⊇ ORM;②ALTER 加列在位(LCQ 两列/messages 快照列);
③中文 ENUM 值面逐字一致(ch06 式)。摘文件→全红的双证记 dev-notes。
"""

import re
from pathlib import Path

from app.db.models import EvalRuns, LowConfidenceQuestion, Message, ReviewQueue

INIT_DIR = Path(__file__).resolve().parent.parent / "db" / "init"

_KEY_PREFIXES = {"PRIMARY", "UNIQUE", "KEY", "CONSTRAINT", "INDEX", "FULLTEXT"}


def _all_sql_text() -> str:
    return "\n".join(sorted(p.read_text(encoding="utf-8")
                             for p in INIT_DIR.glob("*.sql")))


def _strip_quotes(text: str) -> str:
    """引号串整体塌成 ''(COMMENT 里带分号会截 ALTER 块,结构解析前先摘除)。"""
    return re.sub(r"'(?:[^'\\]|\\.|'')*'", "''", text)


def _create_columns(text: str, table: str) -> set[str]:
    m = re.search(
        rf"CREATE TABLE (?:IF NOT EXISTS )?{table}\s*\((.*?)\n\)\s*ENGINE",
        text, re.I | re.S,
    )
    if m is None:
        return set()
    cols = set()
    for line in m.group(1).splitlines():
        km = re.match(r"\s*`?(\w+)`?\s+", line)
        if km and km.group(1).upper() not in _KEY_PREFIXES:
            cols.add(km.group(1))
    return cols


def _alter_added(text: str, table: str) -> set[str]:
    cols = set()
    for block in re.finditer(rf"ALTER TABLE {table}\b(.*?);", text, re.I | re.S):
        cols |= set(re.findall(r"ADD COLUMN\s+`?(\w+)`?", block.group(1), re.I))
    return cols


def _enum_values(text: str, marker: str) -> tuple[str, ...]:
    """行锚抓 `marker ENUM('a','b','c')` 的字面值序(逐字,含中文;防 tool_source 误配)。"""
    m = re.search(rf"(?m)^[\s`]*{marker}[\s`]*ENUM\(([^)]*)\)", text)
    if m is None:
        return ()
    return tuple(re.findall(r"'([^']*)'", m.group(1)))


def test_review_queue_columns_covered_by_ddl():
    ddl = _create_columns(_all_sql_text(), "review_queue")
    model = {c.name for c in ReviewQueue.__table__.columns}
    assert ddl, "db/init 里没有 review_queue 的 CREATE TABLE(09 缺位?)"
    assert model <= ddl, f"模型列在 DDL 里没有: {model - ddl}"


def test_eval_runs_columns_covered_by_ddl():
    ddl = _create_columns(_all_sql_text(), "eval_runs")
    model = {c.name for c in EvalRuns.__table__.columns}
    assert ddl, "db/init 里没有 eval_runs 的 CREATE TABLE(09 缺位?)"
    assert model <= ddl, f"模型列在 DDL 里没有: {model - ddl}"


def test_lcq_and_messages_new_columns_in_alter_union():
    text = _strip_quotes(_all_sql_text())
    lcq = _alter_added(text, "low_confidence_questions")
    missing = {"retrieved_chunks", "matched_review_id"} - lcq
    assert not missing, f"LCQ 飞轮两列不在 ALTER 并集(09 缺位?): {missing}"
    msg = _alter_added(text, "messages")
    assert "retrieval_snapshot" in msg, "messages 快照列缺位(10_ 丢了?)"


def test_chinese_enum_values_verbatim_in_ddl():
    text = _all_sql_text()
    assert _enum_values(text, "review_status") == ("待审", "通过", "驳回"), \
        "review_queue.review_status 值面漂移(审核状态机的字面根基)"
    assert _enum_values(text, "triggered_by") == ("定时", "手动"), \
        "eval_runs.triggered_by 值面漂移"


def test_lcq_source_enum_union_covers_user_feedback():
    """👎 入口启用 user_feedback 空位——DDL 值面必须含它(ch04 原文即含,防改写)。"""
    vals = _enum_values(_all_sql_text(), "source")
    assert "user_feedback" in vals, f"lcq_source 缺 user_feedback: {vals}"
