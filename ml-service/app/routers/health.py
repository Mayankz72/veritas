from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Document as DocumentModel
from app.schemas.document import SectionHealth
from app.services.health import compute_document_health

router = APIRouter(prefix="/documents/{document_id}/health", tags=["health"])


@router.get("", response_model=list[SectionHealth])
def get_document_health(document_id: str, db: Session = Depends(get_db)) -> list[SectionHealth]:
    document = db.get(DocumentModel, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    stats = compute_document_health(db, document_id)
    result = []
    for s in stats:
        flagged, reason = s.flag()
        result.append(
            SectionHealth(
                section=s.section,
                claim_count=s.claim_count,
                supported_count=s.supported_count,
                partial_count=s.partial_count,
                unsupported_count=s.unsupported_count,
                not_checked_count=s.not_checked_count,
                average_grounding_score=s.average_grounding_score,
                flagged=flagged,
                flag_reason=reason,
            )
        )
    return result
