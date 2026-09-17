import re
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Claim as ClaimModel
from app.models import Document as DocumentModel
from app.models import Publication as PublicationModel
from app.models import QuizQuestion as QuizQuestionModel
from app.routers.claims import _to_schema as claim_to_schema
from app.routers.documents import _document_to_schema
from app.routers.quiz import _to_schema as quiz_to_schema
from app.schemas.document import Publication, PublicationBundle, VeritasExport

documents_router = APIRouter(prefix="/documents/{document_id}/publish", tags=["publications"])
publications_router = APIRouter(prefix="/publications", tags=["publications"])


class PublishRequest(BaseModel):
    expires_in_hours: int | None = None
    include_figures: bool = True


def _to_schema(pub: PublicationModel) -> Publication:
    return Publication(
        id=pub.id,
        document_id=pub.document_id,
        include_figures=pub.include_figures,
        expires_at=pub.expires_at,
        created_at=pub.created_at,
    )


def _get_live_publication(db: Session, publication_id: str) -> PublicationModel:
    publication = db.get(PublicationModel, publication_id)
    if publication is None:
        raise HTTPException(status_code=404, detail="Publication not found")
    if publication.expires_at is not None and publication.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=410, detail="This published link has expired")
    return publication


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "document"


@documents_router.post("", response_model=Publication)
def publish_document(
    document_id: str, payload: PublishRequest, db: Session = Depends(get_db)
) -> Publication:
    document = db.get(DocumentModel, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    expires_at = None
    if payload.expires_in_hours is not None:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=payload.expires_in_hours)

    publication = PublicationModel(
        document_id=document_id,
        include_figures=payload.include_figures,
        expires_at=expires_at,
    )
    db.add(publication)
    db.commit()

    return _to_schema(publication)


@publications_router.get("/{publication_id}", response_model=PublicationBundle)
def get_publication(publication_id: str, db: Session = Depends(get_db)) -> PublicationBundle:
    publication = _get_live_publication(db, publication_id)
    document = db.get(DocumentModel, publication.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document no longer exists")

    claims = (
        db.execute(
            select(ClaimModel).where(
                ClaimModel.document_id == document.id, ClaimModel.is_current.is_(True)
            )
        )
        .scalars()
        .all()
    )
    quiz = (
        db.execute(select(QuizQuestionModel).where(QuizQuestionModel.document_id == document.id))
        .scalars()
        .all()
    )

    return PublicationBundle(
        publication=_to_schema(publication),
        document=_document_to_schema(document),
        claims=[claim_to_schema(c) for c in claims],
        quiz=[quiz_to_schema(q) for q in quiz],
    )


@publications_router.get("/{publication_id}/export")
def export_publication(publication_id: str, db: Session = Depends(get_db)) -> Response:
    publication = _get_live_publication(db, publication_id)
    document = db.get(DocumentModel, publication.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document no longer exists")

    claims = (
        db.execute(
            select(ClaimModel).where(
                ClaimModel.document_id == document.id, ClaimModel.is_current.is_(True)
            )
        )
        .scalars()
        .all()
    )
    quiz = (
        db.execute(select(QuizQuestionModel).where(QuizQuestionModel.document_id == document.id))
        .scalars()
        .all()
    )

    export = VeritasExport(
        publication_id=publication.id,
        exported_at=datetime.now(timezone.utc),
        document=_document_to_schema(document),
        claims=[claim_to_schema(c) for c in claims],
        quiz=[quiz_to_schema(q) for q in quiz],
    )
    filename = f"{_slugify(document.title)}.veritas.json"
    return Response(
        content=export.model_dump_json(indent=2, by_alias=True),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
