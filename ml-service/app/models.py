import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

EMBEDDING_DIM = 384


def _uuid() -> str:
    return uuid.uuid4().hex


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Chunk.order_index"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str | None] = mapped_column(String, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Float), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)
    section: Mapped[str | None] = mapped_column(String, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_score: Mapped[float] = mapped_column(Float, nullable=False)
    grounding_label: Mapped[str | None] = mapped_column(String, nullable=True)
    grounding_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class QuizQuestion(Base):
    __tablename__ = "quiz_questions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)
    section: Mapped[str | None] = mapped_column(String, nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(String, nullable=False)
    source_chunk_id: Mapped[str] = mapped_column(String, nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    grounding_label: Mapped[str | None] = mapped_column(String, nullable=True)
    grounding_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
