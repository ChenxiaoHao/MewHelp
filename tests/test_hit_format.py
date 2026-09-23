from types import SimpleNamespace

from app.rag.hit_format import build_citations, format_hits
from app.rag.retriever import ScoredRow


def _row(i):
    return SimpleNamespace(questions=f"q{i}首行\nq{i}次行", answer=f"a{i}", category=f"c{i}",
                           section_path=f"手册 > 节{i}")


def test_format_hits_numbers_and_first_line():
    hits = format_hits([ScoredRow(7, 0.9, _row(7)), ScoredRow(9, 0.5, _row(9))])
    assert hits[0] == {"n": 1, "id": 7, "question": "q7首行", "answer": "a7",
                       "category": "c7", "section_path": "手册 > 节7"}
    assert hits[1]["n"] == 2 and hits[1]["question"] == "q9首行"


def test_build_citations_maps_and_filters_legacy():
    hits = format_hits([ScoredRow(7, 0.9, _row(7))]) + [{"id": 1, "question": "旧", "answer": "旧", "category": "旧"}]
    cites = build_citations(hits)
    assert cites == [{"n": 1, "chunk_id": 7, "section_path": "手册 > 节7", "question": "q7首行", "answer": "a7"}]
    assert build_citations([]) == []
