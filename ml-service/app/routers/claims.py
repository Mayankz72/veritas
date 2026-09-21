import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Claim as ClaimModel
from app.models import Document as DocumentModel
from app.schemas.document import GroundedClaim
from app.services.claim_generation import extract_supporting_quote, get_provider
from app.services.grounding_verifier import get_verifier
from app.services.retrieval import retrieve_chunks

router = APIRouter(prefix="/documents/{document_id}/claims", tags=["claims"])


class GenerateClaimsRequest(BaseModel):
    query: str
    section: str | None = None
    k: int = 5
    max_claims: int = 5


def _to_schema(claim: ClaimModel) -> GroundedClaim:
    return GroundedClaim(
        id=claim.id,
        document_id=claim.document_id,
        section=claim.section,
        text=claim.text,
        source_chunk_ids=claim.source_chunk_ids,
        page=claim.page,
        quote=claim.quote,
        retrieval_score=claim.retrieval_score,
        grounding_label=claim.grounding_label,
        grounding_score=claim.grounding_score,
        version=claim.version,
        is_current=claim.is_current,
    )


def _build_claims(
    db: Session,
    document_id: str,
    query: str,
    section: str | None,
    k: int,
    max_claims: int,
    version: int,
) -> list[ClaimModel]:
    retrieved = retrieve_chunks(db, document_id, query, k=k)
    if not retrieved:
        raise HTTPException(
            status_code=422,
            detail="No embedded chunks found for this document - was it fully ingested?",
        )

    passages = [rc.chunk.text for rc in retrieved]
    provider = get_provider()
    generated = provider.generate_claims(query, passages, max_claims=max_claims)
    verifier = get_verifier()

    claim_models: list[ClaimModel] = []
    for gen_claim in generated:
        source = retrieved[gen_claim.source_chunk_index]
        quote = extract_supporting_quote(gen_claim.text, source.chunk.text)

        grounding_label, grounding_score = None, None
        if verifier is not None:
            grounding_label, grounding_score = verifier.predict(gen_claim.text, source.chunk.text)

        claim_models.append(
            ClaimModel(
                id=uuid.uuid4().hex,
                document_id=document_id,
                section=section,
                text=gen_claim.text,
                source_chunk_ids=[source.chunk.id],
                page=source.chunk.page,
                quote=quote,
                retrieval_score=source.score,
                grounding_label=grounding_label,
                grounding_score=grounding_score,
                version=version,
                is_current=True,
            )
        )
    return claim_models


@router.post("/generate", response_model=list[GroundedClaim])
def generate_claims(
    document_id: str, payload: GenerateClaimsRequest, db: Session = Depends(get_db)
) -> list[GroundedClaim]:
    document = db.get(DocumentModel, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    claim_models = _build_claims(
        db, document_id, payload.query, payload.section, payload.k, payload.max_claims, version=1
    )
    db.add_all(claim_models)
    db.commit()

    return [_to_schema(c) for c in claim_models]


@router.post("/regenerate", response_model=list[GroundedClaim])
def regenerate_claims(
    document_id: str, payload: GenerateClaimsRequest, db: Session = Depends(get_db)
) -> list[GroundedClaim]:
    """Supersedes the current claims for a section with a fresh generation,
    keeping prior versions in the database (is_current=False) rather than
    deleting them - a lightweight version history."""
    document = db.get(DocumentModel, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    previous_version = db.execute(
        select(func.max(ClaimModel.version)).where(
            ClaimModel.document_id == document_id, ClaimModel.section == payload.section
        )
    ).scalar()
    next_version = (previous_version or 0) + 1

    claim_models = _build_claims(
        db,
        document_id,
        payload.query,
        payload.section,
        payload.k,
        payload.max_claims,
        version=next_version,
    )

    current_claims = (
        db.execute(
            select(ClaimModel).where(
                ClaimModel.document_id == document_id,
                ClaimModel.section == payload.section,
                ClaimModel.is_current.is_(True),
            )
        )
        .scalars()
        .all()
    )
    for claim in current_claims:
        claim.is_current = False

    db.add_all(claim_models)
    db.commit()

    return [_to_schema(c) for c in claim_models]


@router.get("", response_model=list[GroundedClaim])
def list_claims(
    document_id: str, current_only: bool = True, db: Session = Depends(get_db)
) -> list[GroundedClaim]:
    query = select(ClaimModel).where(ClaimModel.document_id == document_id)
    if current_only:
        query = query.where(ClaimModel.is_current.is_(True))
    claims = db.execute(query).scalars().all()
    return [_to_schema(c) for c in claims]
