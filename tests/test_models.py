from sqlalchemy import JSON


def test_conversation_table():
    from app.db.models import Conversation

    t = Conversation.__table__
    assert t.name == "conversations"
    assert t.c.id.primary_key
    assert t.c.user_id.type.length == 64
    assert list(t.c.status.type.enums) == ["进行中", "已转人工", "已结束"]
    assert {"created_at", "updated_at"} <= set(t.c.keys())


def test_message_table():
    from app.db.models import Message

    t = Message.__table__
    assert t.name == "messages"
    assert list(t.c.role.type.enums) == ["user", "assistant", "tool"]
    assert isinstance(t.c.tool_calls.type, JSON)
    assert t.c.content.nullable
    fks = {fk.target_fullname for fk in t.c.conversation_id.foreign_keys}
    assert "conversations.id" in fks


def test_faq_table():
    from app.db.models import Faq

    t = Faq.__table__
    assert t.name == "faq"
    assert t.c.question.type.length == 512
    assert t.c.answer.nullable is False
    assert t.c.category.index


def test_ticket_table():
    from app.db.models import Ticket

    t = Ticket.__table__
    assert t.name == "tickets"
    assert t.c.ticket_no.primary_key
    assert t.c.ticket_no.type.length == 32
    assert list(t.c.ticket_type.type.enums) == ["售后", "投诉", "咨询"]
    assert list(t.c.status.type.enums) == ["待处理", "已处理"]
    fks = {fk.target_fullname for fk in t.c.conversation_id.foreign_keys}
    assert "conversations.id" in fks
