from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Claim as ClaimModel
from app.models import Topic, TopicPaper
from app.schemas.topic import (
    ArxivCandidate,
    BriefItem,
    CreateTopicRequest,
    PaperBrief,
    SearchTopicRequest,
    TopicOut,
    TopicPaperOut,
    TopicSummary,
    TopicSynthesis,
)
from app.services.paper_brief import ASPECT_TITLES, ASPECTS
from app.services.topic_pipeline import refresh_topic_synthesis, run_topic_pipeline
from app.services.topic_search import find_candidates

router = APIRouter(prefix="/topics", tags=["topics"])

_TITLE_TO_KEY = {a.title: a.key for a in ASPECTS}
MAX_PAPERS = 8


def _brief_for(db: Session, document_id: str | None) -> PaperBrief:
    brief: dict[str, list[BriefItem]] = {a.key: [] for a in ASPECTS}
    if document_id:
        claims = db.execute(
            select(ClaimModel)
            .where(
                ClaimModel.document_id == document_id,
                ClaimModel.section.in_(ASPECT_TITLES),
                ClaimModel.is_current.is_(True),
            )
            .order_by(ClaimModel.created_at)
        ).scalars()
        for claim in claims:
            brief[_TITLE_TO_KEY[claim.section]].append(
                BriefItem(
                    id=claim.id,
                    text=claim.text,
                    quote=claim.quote,
                    page=claim.page,
                    retrieval_score=claim.retrieval_score,
                    grounding_label=claim.grounding_label,
                    grounding_score=claim.grounding_score,
                )
            )
    return PaperBrief(**brief)


def _to_schema(db: Session, topic: Topic) -> TopicOut:
    return TopicOut(
        id=topic.id,
        query=topic.query,
        status=topic.status,
        error=topic.error,
        created_at=topic.created_at,
        papers=[
            TopicPaperOut(
                id=p.id,
                arxiv_id=p.arxiv_id,
                title=p.title,
                authors=[a for a in p.authors.split(", ") if a],
                published=p.published,
                abstract=p.abstract,
                status=p.status,
                error=p.error,
                document_id=p.document_id,
                brief_mode=p.brief_mode,
                brief=_brief_for(db, p.document_id),
            )
            for p in topic.papers
        ],
        synthesis=TopicSynthesis.model_validate(topic.synthesis) if topic.synthesis else None,
    )


@router.post("/search", response_model=list[ArxivCandidate])
def search_topic(payload: SearchTopicRequest) -> list[ArxivCandidate]:
    """Preview the papers a topic would analyse, so the user can pick."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Enter a topic to search for.")
    try:
        candidates = find_candidates(query, limit=min(max(payload.limit, 1), MAX_PAPERS))
    except Exception as exc:  # noqa: BLE001 - arXiv down / rate-limited
        raise HTTPException(status_code=502, detail=f"arXiv search failed: {exc}") from exc
    return [
        ArxivCandidate(
            arxiv_id=c.paper.arxiv_id,
            title=c.paper.title,
            authors=c.paper.authors,
            published=c.paper.published,
            abstract=c.paper.abstract,
            score=c.score,
            is_recent=c.is_recent,
        )
        for c in candidates
    ]


@router.post("", response_model=TopicSummary)
def create_topic(
    payload: CreateTopicRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TopicSummary:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Enter a topic to search for.")
    max_papers = min(max(payload.max_papers, 1), MAX_PAPERS)
    papers = payload.papers[:MAX_PAPERS] if payload.papers else None

    topic = Topic(query=query, status="searching")
    db.add(topic)
    db.commit()

    background_tasks.add_task(run_topic_pipeline, topic.id, papers, max_papers)
    return TopicSummary(
        id=topic.id,
        query=topic.query,
        status=topic.status,
        paper_count=len(papers or []),
        created_at=topic.created_at,
    )


@router.get("", response_model=list[TopicSummary])
def list_topics(db: Session = Depends(get_db)) -> list[TopicSummary]:
    rows = db.execute(
        select(Topic, func.count(TopicPaper.id))
        .outerjoin(TopicPaper, TopicPaper.topic_id == Topic.id)
        .group_by(Topic.id)
        .order_by(Topic.created_at.desc())
        .limit(20)
    ).all()
    return [
        TopicSummary(
            id=t.id, query=t.query, status=t.status, paper_count=count, created_at=t.created_at
        )
        for t, count in rows
    ]


@router.get("/{topic_id}", response_model=TopicOut)
def get_topic(topic_id: str, db: Session = Depends(get_db)) -> TopicOut:
    topic = db.get(Topic, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    return _to_schema(db, topic)


@router.delete("/{topic_id}", status_code=204)
def delete_topic(topic_id: str, db: Session = Depends(get_db)) -> None:
    """Removes a topic and its paper rows. The parsed documents and their briefs
    are kept: they are the cache that lets the same paper be reused by another
    topic without being downloaded and embedded again."""
    topic = db.get(Topic, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    db.delete(topic)
    db.commit()


@router.post("/{topic_id}/refresh", response_model=TopicSummary)
def refresh_topic(
    topic_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> TopicSummary:
    """Recompute the cross-paper synthesis from the current per-paper briefs."""
    topic = db.get(Topic, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    background_tasks.add_task(refresh_topic_synthesis, topic_id)
    return TopicSummary(
        id=topic.id,
        query=topic.query,
        status=topic.status,
        paper_count=len(topic.papers),
        created_at=topic.created_at,
    )
