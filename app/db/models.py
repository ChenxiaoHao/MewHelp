"""ORM 模型：四张表与 spec 附录 A 的 DDL（用户原文）逐列对齐。

建表不走 metadata.create_all——以 db/init/01_schema.sql（用户 DDL 原样）为准，
容器首启自动执行；本文件仅供查询/写入映射与结构测试。
"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Enum, ForeignKey, String, Text, func, text
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _pk() -> Mapped[int]:
    return mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = _pk()
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束", name="conversation_status"),
        nullable=False,
        server_default="进行中",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = _pk()
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("conversations.id"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        Enum("user", "assistant", "tool", name="message_role"), nullable=False
    )
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = _pk()
    question: Mapped[str] = mapped_column(String(512), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("conversations.id"),
        nullable=False,
        index=True,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(
        Enum("售后", "投诉", "咨询", name="ticket_type"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="ticket_status"),
        nullable=False,
        server_default="待处理",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class KnowledgeChunk(Base):
    """知识库原文权威源;DDL 与 spec 附录 A 逐列对齐,建表以 03_ch03_schema.sql 为准。"""

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = _pk()
    category: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[str | None] = mapped_column(String(512))
    content_type: Mapped[str | None] = mapped_column(String(32))
    is_key_clause: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("0"))
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"),
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"),
    )
    vector_id: Mapped[str | None] = mapped_column(String(64))
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done", name="vectorize_status"),
        nullable=False,
        server_default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class QaExtractionStaging(Base):
    """历史对话抽 QA 的中转表;保留行可追溯,--clear-staging 才物理清。"""

    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = _pk()
    batch_no: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_ref: Mapped[str | None] = mapped_column(String(255))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded", name="staging_status"),
        nullable=False,
        server_default="extracted",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
