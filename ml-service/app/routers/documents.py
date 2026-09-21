from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Document as DocumentModel
from app.schemas.document import Chunk as ChunkSchema
from app.schemas.document import ParsedDocument
from app.schemas.document import RelatedDocument as RelatedDocumentSchema
from app.services.arxiv import InvalidArxivId, fetch_arxiv_pdf, normalize_arxiv_id
from app.services.ingestion import document_embedding as _document_embedding  # noqa: F401
from app.services.ingestion import persist_document
from app.services.pdf_extraction import extract_document
from app.services.related_documents import find_related

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
    return _document_to_schema(persist_document(db, source_type, title, page_count, chunks))


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


@router.get("/{document_id}/related", response_model=list[RelatedDocumentSchema])
def get_related_documents(
    document_id: str, k: int = 5, db: Session = Depends(get_db)
) -> list[RelatedDocumentSchema]:
    document = db.get(DocumentModel, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    related = find_related(db, document_id, k=k)
    return [
        RelatedDocumentSchema(document_id=r.document.id, title=r.document.title, score=r.score)
        for r in related
    ]
