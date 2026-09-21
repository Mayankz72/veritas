from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.schemas.document import CamelModel

GroundingLabel = Literal["supported", "unsupported", "partial"]


class ArxivCandidate(CamelModel):
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    abstract: str
    score: float  # similarity of the paper's title + abstract to the topic
    is_recent: bool = False  # published within the last two years


class BriefItem(CamelModel):
    id: str
    text: str
    quote: str
    page: int
    retrieval_score: float
    grounding_label: GroundingLabel | None
    grounding_score: float | None


class PaperBrief(CamelModel):
    problem: list[BriefItem] = []
    method: list[BriefItem] = []
    results: list[BriefItem] = []
    matters: list[BriefItem] = []


class TopicPaperOut(CamelModel):
    id: str
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    abstract: str
    status: Literal["pending", "ingesting", "analyzing", "done", "error"]
    error: str | None
    document_id: str | None
    brief_mode: str | None
    brief: PaperBrief


class TimelineEntry(CamelModel):
    paper_id: str
    label: str
    title: str
    year: str
    published: str


class RelationshipOut(CamelModel):
    from_paper_id: str
    to_paper_id: str
    kind: Literal["cites", "similar"]
    explanation: str
    page: int | None = None
    snippet: str | None = None
    shared_terms: list[str] = []
    similarity: float | None = None


class HowTheyFit(CamelModel):
    paper_ids: list[str]
    text: str


class ComparisonRow(CamelModel):
    paper_id: str
    label: str
    title: str
    year: str
    problem: str
    method: str
    results: str
    matters: str


class TopicSynthesis(CamelModel):
    mode: str  # "template" or "llm:<provider>"
    overview: str
    reading_order: list[str]
    timeline: list[TimelineEntry]
    relationships: list[RelationshipOut]
    how_they_fit: list[HowTheyFit]
    comparison: list[ComparisonRow]
    foundational: list[str]


class TopicOut(CamelModel):
    id: str
    query: str
    status: Literal["searching", "ingesting", "analyzing", "synthesizing", "done", "error"]
    error: str | None
    created_at: datetime
    papers: list[TopicPaperOut]
    synthesis: TopicSynthesis | None


class TopicSummary(CamelModel):
    id: str
    query: str
    status: str
    paper_count: int
    created_at: datetime


# ---- request bodies (snake_case, like the other routers' request models) ----


class PaperMeta(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str] = []
    published: str = ""
    abstract: str = ""


class SearchTopicRequest(BaseModel):
    query: str
    limit: int = 5


class CreateTopicRequest(BaseModel):
    query: str
    max_papers: int = 5
    # Papers the user picked from a search preview; omitted = auto-pick the top matches.
    papers: list[PaperMeta] | None = None
