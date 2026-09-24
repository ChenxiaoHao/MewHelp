import pytest
from pydantic import ValidationError


def test_conversation_id_optional_and_defaults_none():
    from app.schemas.chat import ChatRequest

    r = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    assert r.conversation_id is None


def test_conversation_id_accepts_positive_int():
    from app.schemas.chat import ChatRequest

    r = ChatRequest(messages=[{"role": "user", "content": "hi"}], conversation_id=7)
    assert r.conversation_id == 7


def test_conversation_id_rejects_zero():
    from app.schemas.chat import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "user", "content": "hi"}], conversation_id=0)


def test_event_models_match_spec_frame_shape():
    """spec §6 帧 data 形状契约。"""
    from app.schemas.chat import ConversationEvent, ToolCallEvent, ToolResultEvent

    assert ConversationEvent(conversation_id=3).model_dump() == {"conversation_id": 3}
    assert ToolCallEvent(
        id="call_x", name="query_logistics", args={"order_id": "1001"}
    ).model_dump() == {"id": "call_x", "name": "query_logistics", "args": {"order_id": "1001"}}
    assert ToolResultEvent(
        id="call_x", name="query_logistics", ok=True, summary="运输中"
    ).model_dump(exclude_none=True) == {"id": "call_x", "name": "query_logistics", "ok": True, "summary": "运输中"}
    # ch04: citations 可选键——不带时帧形状逐字符不变(兼容红线),带时追加
    assert "citations" not in ToolResultEvent(
        id="c", name="query_faq", ok=True, summary="s").model_dump(exclude_none=True)
    assert ToolResultEvent(id="c", name="query_faq", ok=True, summary="s",
                           citations=[{"n": 1, "chunk_id": 7}]
                           ).model_dump(exclude_none=True)["citations"] == [{"n": 1, "chunk_id": 7}]
