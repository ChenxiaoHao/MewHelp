"""ch07 T2 接缝:db/init DDL 并集 ⊇ ORM 列(spec「接缝测试扩展」)。

docker compose 首启按 01→…→06→07a→07b 升序建库;若 07 两文件缺失/列名漂移,
新环境重建后锚点读写会「模型有列、库里没列」静默炸(ORM 查询 1054)。本测把这类
结构盲区转成单元断言——ch06 接缝(ENUM 值面)同款思路,这次钉的是列名与表存在性。
演示过「移走 07 文件→本测全红、放回→全绿」双证(见 dev-notes ch07 T2 段)。
"""

import re
from pathlib import Path

from app.db.models import Conversation, ConversationSummary

INIT_DIR = Path(__file__).resolve().parent.parent / "db" / "init"

_KEY_PREFIXES = {"PRIMARY", "UNIQUE", "KEY", "CONSTRAINT", "INDEX", "FULLTEXT"}


def _create_columns(text: str, table: str) -> set[str]:
    """CREATE TABLE <table> (...) 的列名集(跳过约束行);表不存在则空集。"""
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


def _conversations_ddl_cols() -> set[str]:
    """conversations 面 = 01 CREATE ∪ 各文件 ALTER TABLE conversations ADD COLUMN。"""
    cols = set()
    for sql in sorted(INIT_DIR.glob("*.sql")):
        text = sql.read_text(encoding="utf-8")
        cols |= _create_columns(text, "conversations")
        for block in re.finditer(r"ALTER TABLE conversations\b(.*?);", text, re.I | re.S):
            cols |= set(re.findall(r"ADD COLUMN\s+`?(\w+)`?", block.group(1), re.I))
    return cols


def _summaries_ddl_cols() -> set[str]:
    cols = set()
    for sql in sorted(INIT_DIR.glob("*.sql")):
        cols |= _create_columns(sql.read_text(encoding="utf-8"), "conversation_summaries")
    return cols


def test_ch07_anchor_columns_present_in_ddl_union():
    ddl = _conversations_ddl_cols()
    missing = {"summary", "summary_upto_msg_id", "layer1_from_msg_id"} - ddl
    assert not missing, f"db/init 并集缺列(07a/07b 丢了或被改写?): {missing}"


def test_conversation_model_columns_covered_by_ddl():
    ddl = _conversations_ddl_cols()
    model = {c.name for c in Conversation.__table__.columns}
    assert model <= ddl, f"模型列在 DDL 并集里没有 → 新环境重建即 1054: {model - ddl}"


def test_conversation_summaries_table_and_model_aligned():
    ddl = _summaries_ddl_cols()
    assert ddl, "conversation_summaries 表在 db/init 里不存在(07b 丢了?)"
    model = {c.name for c in ConversationSummary.__table__.columns}
    assert model <= ddl, f"段表模型列缺 DDL: {model - ddl}"
    # 约束面(uk_conv_seq 等)由 dev 活库 SHOW CREATE TABLE 取证,单测只钉列名


def test_07_sql_files_exist_verbatim():
    names = {p.name for p in INIT_DIR.glob("*.sql")}
    assert {"07a_ch07_summary_projection.sql", "07b_ch07_layers.sql"} <= names
