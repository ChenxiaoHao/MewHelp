"""池写入 crud:一行 add + 一次 commit。"""

import pytest

from app.db import crud


class FakeSession:
    def __init__(self):
        self.added, self.commits = [], 0

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_add_low_confidence_question():
    from app.db.models import LowConfidenceQuestion

    s = FakeSession()
    await crud.add_low_confidence_question(s, conversation_id=None, raw_question="q?",
                                           source="user_feedback", reason=None)
    assert s.commits == 1 and len(s.added) == 1
    row = s.added[0]
    assert isinstance(row, LowConfidenceQuestion) and row.source == "user_feedback"
