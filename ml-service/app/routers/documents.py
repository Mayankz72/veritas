import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Chunk as ChunkModel
from app.models import Document as DocumentModel
from app.schemas.document import Chunk as ChunkSchema
from app.schemas.document import ParsedDocument
from app.services.arxiv import InvalidArxivId, fetch_arxiv_pdf, normalize_arxiv_id
from app.services.embeddings import embed_texts
from app.services.pdf_extraction import extract_document

router = APIRouter(prefix="/documents", tags=["documents"])


class ArxivIngestRequest(BaseModel):
    arxiv_id: str


def _document_to_schema(document: DocumentModel) -> ParsedDocument:
    return ParsedDocument(
        id=document.id,
        source_type=document.source_type,
        title=document.title,
        page_count=document.page_count,
        chunks=[
            ChunkSchema(
                id=c.id,
                page=c.page,
                section=c.section,
                text=c.text,
                bbox=tuple(c.bbox) if c.bbox else None,
            )
            for c in document.chunks
        ],
    )


def _persist_document(
    db: Session, source_type: str, title: str, page_count: int, chunks
) -> ParsedDocument:
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

    db.commit()

    return _document_to_schema(document)


@router.post("/upload", response_model=ParsedDocument)
async def upload_document(file: UploadFile, db: Session = Depends(get_db)) -> ParsedDocument:
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    title_hint = file.filename.rsplit(".", 1)[0] if file.filename else None
    title, page_count, chunks = extract_document(pdf_bytes, title_hint=title_hint)
    return _persist_document(db, "upload", title, page_count, chunks)


@router.post("/arxiv", response_model=ParsedDocument)
async def ingest_arxiv(
    payload: ArxivIngestRequest, db: Session = Depends(get_db)
) -> ParsedDocument:
    try:
        normalize_arxiv_id(payload.arxiv_id)
    except InvalidArxivId as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pdf_bytes = await fetch_arxiv_pdf(payload.arxiv_id)
    title, page_count, chunks = extract_document(pdf_bytes)
    return _persist_document(db, "arxiv", title, page_count, chunks)


@router.get("/{document_id}", response_model=ParsedDocument)
def get_document(document_id: str, db: Session = Depends(get_db)) -> ParsedDocument:
    document = db.get(DocumentModel, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return _document_to_schema(document)
