from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk
from app.services.embeddings import embed_query


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1], higher is more relevant


def retrieve_chunks(db: Session, document_id: str, query: str, k: int = 5) -> list[RetrievedChunk]:
    query_vector = embed_query(query)
    distance = Chunk.embedding.cosine_distance(query_vector)

    rows = (
        db.execute(
            select(Chunk, distance.label("distance"))
            .where(Chunk.document_id == document_id, Chunk.embedding.is_not(None))
            .order_by(distance)
            .limit(k)
        )
        .tuples()
        .all()
    )
    return [RetrievedChunk(chunk=chunk, score=1.0 - dist) for chunk, dist in rows]
