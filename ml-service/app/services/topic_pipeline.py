"""Background job behind POST /topics: search -> ingest -> brief -> synthesize.

Runs off the request thread and writes progress to the database after every
step, so the UI can poll a topic and show "paper 3 of 5: reading the PDF"
instead of holding one long request open (the failure mode that made
first-time ingestion flaky on the free hosting tier).
"""

from __future__ import annotations

import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Chunk, Document, Topic, TopicPaper
from app.models import Claim as ClaimModel
from app.schemas.topic import PaperMeta
from app.services.arxiv import download_arxiv_pdf
from app.services.claim_generation import LLMProvider, get_provider
from app.services.grounding_verifier import GroundingVerifier, get_verifier
from app.services.ingestion import persist_document
from app.services.paper_brief import ASPECT_TITLES, ASPECTS, build_brief
from app.services.pdf_extraction import extract_document
from app.services.topic_search import find_candidates
from app.services.topic_synthesis import PaperInput, build_synthesis

ARXIV_POLITE_DELAY_SECONDS = 1.5
IN_PROGRESS_STATUSES = ("searching", "ingesting", "analyzing", "synthesizing")
_TITLE_TO_KEY = {a.title: a.key for a in ASPECTS}


def mark_interrupted_topics(db: Session) -> None:
    """A server restart kills background jobs mid-flight; without this the
    topic would report "ingesting" forever."""
    for topic in db.execute(select(Topic).where(Topic.status.in_(IN_PROGRESS_STATUSES))).scalars():
        topic.status = "error"
        topic.error = "Interrupted by a server restart - start the analysis again."
    db.commit()


def _fail_topic(db: Session, topic: Topic, message: str) -> None:
    db.rollback()
    topic.status = "error"
    topic.error = message[:500]
    db.commit()


def _brief_exists(db: Session, document_id: str) -> bool:
    return (
        db.execute(
            select(ClaimModel.id)
            .where(
                ClaimModel.document_id == document_id,
                ClaimModel.section.in_(ASPECT_TITLES),
                ClaimModel.is_current.is_(True),
            )
            .limit(1)
        ).first()
        is not None
    )


def _process_paper(
    db: Session, row: TopicPaper, provider: LLMProvider, verifier: GroundingVerifier | None
) -> None:
    try:
        row.status = "ingesting"
        db.commit()

        # The same arXiv paper analysed for an earlier topic is reused, not re-parsed.
        earlier = db.execute(
            select(TopicPaper)
            .where(
                TopicPaper.arxiv_id == row.arxiv_id,
                TopicPaper.id != row.id,
                TopicPaper.document_id.is_not(None),
                TopicPaper.status == "done",
            )
            .limit(1)
        ).scalar_one_or_none()

        if earlier is not None:
            row.document_id = earlier.document_id
        else:
            time.sleep(ARXIV_POLITE_DELAY_SECONDS)
            pdf_bytes = download_arxiv_pdf(row.arxiv_id)
            title, page_count, chunks = extract_document(pdf_bytes, title_hint=row.title)
            document = persist_document(db, "arxiv", row.title or title, page_count, chunks)
            row.document_id = document.id

        row.status = "analyzing"
        db.commit()

        if earlier is not None and _brief_exists(db, row.document_id):
            row.brief_mode = earlier.brief_mode
        else:
            _, mode = build_brief(
                db, row.document_id, row.title, provider, verifier, abstract=row.abstract
            )
            row.brief_mode = mode

        row.status = "done"
        db.commit()
    except Exception as exc:  # noqa: BLE001 - one bad PDF must not sink the whole topic
        db.rollback()
        row.status = "error"
        row.error = f"{type(exc).__name__}: {exc}"[:500]
        db.commit()


def _paper_inputs(db: Session, rows: list[TopicPaper]) -> list[PaperInput]:
    inputs: list[PaperInput] = []
    for number, row in enumerate(rows, start=1):
        chunks = db.execute(
            select(Chunk.page, Chunk.text)
            .where(Chunk.document_id == row.document_id)
            .order_by(Chunk.order_index)
        ).all()
        claims = db.execute(
            select(ClaimModel)
            .where(
                ClaimModel.document_id == row.document_id,
                ClaimModel.section.in_(ASPECT_TITLES),
                ClaimModel.is_current.is_(True),
            )
            .order_by(ClaimModel.created_at)
        ).scalars()
        brief: dict[str, list[str]] = {a.key: [] for a in ASPECTS}
        for claim in claims:
            brief[_TITLE_TO_KEY[claim.section]].append(claim.text)
        document = db.get(Document, row.document_id)
        inputs.append(
            PaperInput(
                paper_id=row.id,
                label=f"P{number}",
                title=row.title,
                arxiv_id=row.arxiv_id,
                published=row.published,
                abstract=row.abstract,
                chunks=[(page, text) for page, text in chunks],
                embedding=list(document.embedding) if document.embedding is not None else None,
                brief=brief,
            )
        )
    return inputs


def run_topic_pipeline(topic_id: str, papers: list[PaperMeta] | None, max_papers: int) -> None:
    with SessionLocal() as db:
        topic = db.get(Topic, topic_id)
        if topic is None:
            return
        try:
            if papers is None:
                candidates = find_candidates(topic.query, limit=max_papers)
                if not candidates:
                    return _fail_topic(db, topic, "No papers found on arXiv for this topic.")
                papers = [
                    PaperMeta(
                        arxiv_id=c.paper.arxiv_id,
                        title=c.paper.title,
                        authors=c.paper.authors,
                        published=c.paper.published,
                        abstract=c.paper.abstract,
                    )
                    for c in candidates
                ]

            rows = [
                TopicPaper(
                    topic_id=topic.id,
                    position=position,
                    arxiv_id=p.arxiv_id,
                    title=p.title,
                    authors=", ".join(p.authors),
                    published=p.published,
                    abstract=p.abstract,
                    status="pending",
                )
                for position, p in enumerate(papers)
            ]
            db.add_all(rows)
            topic.status = "ingesting"
            db.commit()

            provider, verifier = get_provider(), get_verifier()
            for row in rows:
                _process_paper(db, row, provider, verifier)

            done = [row for row in rows if row.status == "done"]
            if not done:
                return _fail_topic(
                    db, topic, "None of the papers could be processed - see each paper's error."
                )

            topic.status = "synthesizing"
            db.commit()
            topic.synthesis = build_synthesis(topic.query, _paper_inputs(db, done), provider)
            topic.status = "done"
            db.commit()
        except Exception as exc:  # noqa: BLE001
            _fail_topic(db, topic, f"{type(exc).__name__}: {exc}")


def refresh_topic_synthesis(topic_id: str) -> None:
    """Recompute the cross-paper synthesis from the papers' *current* briefs
    (no downloads, no re-parsing) - e.g. after the brief pipeline improved."""
    with SessionLocal() as db:
        topic = db.get(Topic, topic_id)
        if topic is None:
            return
        rows = [p for p in topic.papers if p.status == "done" and p.document_id]
        if not rows:
            return
        try:
            topic.status = "synthesizing"
            db.commit()
            topic.synthesis = build_synthesis(topic.query, _paper_inputs(db, rows), get_provider())
            topic.status = "done"
            db.commit()
        except Exception as exc:  # noqa: BLE001
            _fail_topic(db, topic, f"{type(exc).__name__}: {exc}")
