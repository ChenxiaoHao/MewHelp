def test_normalize_strips_punct_width():
    from app.rag.dedupe import normalize_question

    assert normalize_question("退货运费谁出?") == normalize_question("退货运费 谁出")
    assert normalize_question("ＵＳＢ接口") == "USB接口"  # NFKC 全角转半角


def test_group_exact_merges_literal_duplicates():
    from app.rag.dedupe import group_exact

    rows = [(11, "退货运费谁出"), (12, "包邮门槛是多少"), (13, "退货运费谁出?")]
    assert group_exact(rows) == [[11, 13], [12]]  # 首现顺序;闸1 靶子=C1 双问形态


def test_cosine_values():
    from app.rag.dedupe import cosine

    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([1, 1], [2, 2]) > 0.999


def test_merge_semantic_single_linkage():
    from app.rag.dedupe import merge_semantic

    vecs = {0: [1.0, 0.0], 1: [0.99, 0.14], 2: [0.0, 1.0]}  # 0-1 cosine≈0.99,2 正交
    assert merge_semantic([0, 1, 2], vecs, 0.92) == [[0, 1], [2]]


def test_gate3_split_by_store_similarity():
    from app.rag.dedupe import gate3_split

    def search_for(vec):  # [0.95, ...] 的簇被判重
        return lambda v, k: [(999, 0.0 if v[0] < 0.9 else 0.95)]

    kept, dead = gate3_split([[0], [1]], {0: [1.0, 0.0], 1: [0.0, 1.0]}, search_for(None), 0.92)
    assert kept == [[1]] and dead == [[0]]


async def test_extract_retries_then_skips(monkeypatch):
    """LLM 不合 schema → 重试 1 次 → 再失败整通不落账(source_ref 无记录,重跑自然再抽)。"""
    from app.jobs import mine_qa as job

    class Flaky:
        def __init__(self, fails):
            self.fails, self.calls = fails, 0

        async def ainvoke(self, msgs):
            self.calls += 1
            if self.calls <= self.fails:
                raise ValueError("invalid json")
            from app.schemas.qa_mining import MinedQA, QAItem
            return MinedQA(items=[QAItem(question="问", answer="答")])

    async def fake_transcript(session, cid):
        return "用户: 问\n客服: 答"

    written = []

    async def fake_add(session, batch_no, source_ref, items, status="extracted"):
        written.append(source_ref)
        return len(items)

    class Sess:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    monkeypatch.setattr(job.crud, "conversation_transcript", fake_transcript)
    monkeypatch.setattr(job.crud, "add_qa_staging_rows", fake_add)
    monkeypatch.setattr(job, "get_session_factory", lambda: (lambda: Sess()))

    ok_model = Flaky(fails=1)
    n = await job._extract_one_conversation(ok_model, 9)
    assert n == 1 and written == ["conv:9"]

    bad_model = Flaky(fails=2)
    written.clear()
    n2 = await job._extract_one_conversation(bad_model, 9)
