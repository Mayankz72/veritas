"""Persisting an extracted paper: chunks, embeddings and a whole-document vector.

Shared by the upload/arXiv routes and the topic pipeline, so both ingest a
paper exactly the same way.
"""

import uuid

import numpy as np
from sqlalchemy.orm import Session

from app.models import Chunk as ChunkModel
from app.models import Document as DocumentModel
from app.services.embeddings import embed_texts


def document_embedding(
    chunk_models: list[ChunkModel], embeddings: list[list[float]]
) -> list[float] | None:
    """Whole-document representation for related-paper search
    (app/services/related_documents.py). Averaging *every* chunk's
    embedding dilutes topical signal with references/boilerplate that
    dominate a paper's chunk count - found by comparing results before/
    after: with all-chunk averaging, 5 real ingested papers scored within
    a narrow 0.92-0.96 band regardless of actual relatedness. Using just
    the abstract (or the first few chunks as a fallback when no section is
    labeled "abstract") gives a much more topical fingerprint."""
    abstract_indices = [
        i for i, c in enumerate(chunk_models) if c.section and "abstract" in c.section.lower()
    ]
    indices = abstract_indices or list(range(min(3, len(chunk_models))))
    if not indices:
        return None
    return np.mean(np.array([embeddings[i] for i in indices]), axis=0).tolist()


def persist_document(
    db: Session, source_type: str, title: str, page_count: int, chunks
) -> DocumentModel:
    document = DocumentModel(
        id=uuid.uuid4().hex,
        source_type=source_type,
        title=title,
        page_count=page_count,
    )
    db.add(document)

    chunk_models: list[ChunkModel] = []
    for index, chunk in enumerate(chunks):
        chunk_models.append(
            ChunkModel(
                id=uuid.uuid4().hex,
                document_id=document.id,
                order_index=index,
                page=chunk.page,
                section=chunk.section,
                text=chunk.text,
                bbox=list(chunk.bbox) if chunk.bbox else None,
            )
        )
    db.add_all(chunk_models)

    embeddings = embed_texts([c.text for c in chunk_models])
    for chunk_model, embedding in zip(chunk_models, embeddings, strict=True):
        chunk_model.embedding = embedding

    document.embedding = document_embedding(chunk_models, embeddings)

    db.commit()
    return document
