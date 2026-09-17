from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Base model that (de)serializes as camelCase to match the TS/Zod
    schemas documented in packages/shared/README.md, while staying
    snake_case in Python code."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Chunk(CamelModel):
    id: str
    page: int
    section: str | None
    text: str
    bbox: tuple[float, float, float, float] | None


class ParsedDocument(CamelModel):
    id: str
    source_type: Literal["upload", "arxiv"]
    title: str
    page_count: int
    chunks: list[Chunk]


class GroundedClaim(CamelModel):
    id: str
    document_id: str
    section: str | None
    text: str
    source_chunk_ids: list[str]
    page: int
    quote: str
    retrieval_score: float
    grounding_label: Literal["supported", "unsupported", "partial"] | None
    grounding_score: float | None


class QuizQuestion(CamelModel):
    id: str
    document_id: str
    section: str | None
    question: str
    answer: str
    source_chunk_id: str
    page: int
    grounding_label: Literal["supported", "unsupported", "partial"] | None
    grounding_score: float | None
