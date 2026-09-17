import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Claim as ClaimModel
from app.models import Document as DocumentModel
from app.schemas.document import GroundedClaim
from app.services.claim_generation import extract_supporting_quote, get_provider
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
    )


@router.post("/generate", response_model=list[GroundedClaim])
def generate_claims(
    document_id: str, payload: GenerateClaimsRequest, db: Session = Depends(get_db)
) -> list[GroundedClaim]:
    document = db.get(DocumentModel, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    retrieved = retrieve_chunks(db, document_id, payload.query, k=payload.k)
    if not retrieved:
        raise HTTPException(
            status_code=422,
            detail="No embedded chunks found for this document - was it fully ingested?",
        )

    passages = [rc.chunk.text for rc in retrieved]
    provider = get_provider()
    generated = provider.generate_claims(payload.query, passages, max_claims=payload.max_claims)

    claim_models: list[ClaimModel] = []
    for gen_claim in generated:
        source = retrieved[gen_claim.source_chunk_index]
        claim_models.append(
            ClaimModel(
                id=uuid.uuid4().hex,
                document_id=document_id,
                section=payload.section,
                text=gen_claim.text,
                source_chunk_ids=[source.chunk.id],
                page=source.chunk.page,
                quote=extract_supporting_quote(gen_claim.text, source.chunk.text),
                retrieval_score=source.score,
            )
        )
    db.add_all(claim_models)
    db.commit()

    return [_to_schema(c) for c in claim_models]


@router.get("", response_model=list[GroundedClaim])
def list_claims(document_id: str, db: Session = Depends(get_db)) -> list[GroundedClaim]:
    claims = db.execute(
        select(ClaimModel).where(ClaimModel.document_id == document_id)
    ).scalars().all()
    return [_to_schema(c) for c in claims]
