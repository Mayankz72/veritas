from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document


@dataclass
class RelatedDocument:
    document: Document
    score: float  # cosine similarity in [-1, 1], higher is more related


def find_related(db: Session, document_id: str, k: int = 5) -> list[RelatedDocument]:
    """Nearest-neighbor search over whole-document embeddings (the mean of
    each document's chunk embeddings - see Document.embedding). Meant for a
    small ingested corpus; with only a handful of documents this is a
    qualitative demo of the retrieval mechanism, not a benchmarked
    recall@k system (see evals/related_documents_notes.md)."""
    target = db.get(Document, document_id)
    if target is None or target.embedding is None:
        return []

    distance = Document.embedding.cosine_distance(target.embedding)
    rows = (
        db.execute(
            select(Document, distance.label("distance"))
            .where(Document.id != document_id, Document.embedding.is_not(None))
            .order_by(distance)
            .limit(k)
        )
        .tuples()
        .all()
    )
    return [RelatedDocument(document=doc, score=1.0 - dist) for doc, dist in rows]
