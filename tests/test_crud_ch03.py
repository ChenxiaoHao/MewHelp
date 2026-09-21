"""crud 扩展:FakeSession 走语句/属性级断言;活库行为(TRUNCATE 1701、枚举读写)归 Task 9 集成。"""

import hashlib

import pytest


class FakeResult:
    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class FakeSession:
    def __init__(self, results=None):
        self.added = []
        self.commits = 0
        self.executed = []  # (stmt, FakeResult)
        self._results = list(results or [])
        self._next_id = 100

    def add_all(self, objs):
        self.added.extend(objs)

    async def flush(self):
        for o in self.added:
            if getattr(o, "id", None) is None:
                o.id = self._next_id
                self._next_id += 1

    async def commit(self):
        self.commits += 1

    async def execute(self, stmt):
        res = self._results.pop(0) if self._results else FakeResult()
        self.executed.append(stmt)
        return res


def _mysql_sql(stmt) -> str:
    from sqlalchemy.dialects import mysql

    # literal_binds:形状断言要看字面量('conv:' 与 LIMIT 5),默认编译全变 %s 绑定参数
    return str(stmt.compile(dialect=mysql.dialect(), compile_kwargs={"literal_binds": True}))


def _draft(i: int):
    from app.rag.chunking import ChunkDraft

    return ChunkDraft(category=f"c{i}", questions=f"q{i}", answer=f"a{i}",
                      section_path="p", content_type="policy", is_key_clause=False)


async def test_add_chunk_drafts_pending_and_chained():
    from app.db import crud

    s = FakeSession()
    rows = await crud.add_chunk_drafts(s, [_draft(i) for i in range(3)])
    assert all(r.vectorize_status == "pending" and r.vector_id is None for r in rows)
    assert rows[0].prev_chunk_id is None and rows[0].next_chunk_id == rows[1].id
    assert rows[1].prev_chunk_id == rows[0].id and rows[2].next_chunk_id is None
    assert s.commits == 1


async def test_truncate_wraps_fk_switch():
    """核对点⑤计划语义:session 级 FK 开关 → TRUNCATE → 复原(活库 1701 实测归 Task 9 集成)。"""
    from app.db import crud

    s = FakeSession()
    await crud.truncate_knowledge_chunks(s)
    sqls = [_mysql_sql(e) if not hasattr(e, "text") else e.text for e in s.executed]
    assert any("FOREIGN_KEY_CHECKS" in x and "0" in x for x in sqls[:1]), sqls
    assert any("TRUNCATE" in x.upper() for x in sqls)
    assert any("FOREIGN_KEY_CHECKS" in x and "1" in x for x in sqls[1:]), sqls
    assert s.commits == 1


async def test_mark_chunks_vectorized_backfills_ids():
    from app.db import crud

    s = FakeSession()
    await crud.mark_chunks_vectorized(s, [7, 8])
    assert len([e for e in s.executed if "knowledge_chunks" in _mysql_sql(e)]) == 2
    assert s.commits == 1


def test_unmined_query_shape():
    from app.db import crud

    sql = _mysql_sql(crud.build_unmined_conversations_query(5))
    u = sql.upper()
    assert "CONCAT" in u and "'conv:'" in sql
    assert "NOT IN" in u and "LIMIT" in u and "5" in sql
    assert "MESSAGES" in u and "QA_EXTRACTION_STAGING" in u


async def test_finalize_qa_single_transaction():
    """§6 幂等核心:chunk 插入与 staging 翻面在同一次 commit 前完成。"""
    from app.db import crud

    s = FakeSession()
    kept = [(_draft(1), [11, 12]), (_draft(2), [13])]
    n = await crud.finalize_qa(s, kept, discarded_ids=[14])
    assert n == 2
    assert s.commits == 1  # 只 commit 一次 → 中断必整体回滚
    assert len(s.added) == 2 and {r.content_type for r in s.added} == {"qa_mined"}
    updates = [_mysql_sql(e) for e in s.executed]
    assert sum("qa_extraction_staging" in x and "UPDATE" in x.upper() for x in updates) == 2  # kept+discarded


async def test_add_qa_staging_rows():
    from app.db import crud

    s = FakeSession()
    n = await crud.add_qa_staging_rows(s, "b1", "conv:9", [("问一", "答一"), ("问二", "答二")])
    assert n == 2 and s.commits == 1
    assert {r.source_ref for r in s.added} == {"conv:9"} and all(r.status == "extracted" for r in s.added)


def test_chunk_fingerprint_pure():
    from app.db import crud

    assert crud.chunk_fingerprint("c", "q", "a") == hashlib.sha1("c|q|a".encode("utf-8")).hexdigest()
