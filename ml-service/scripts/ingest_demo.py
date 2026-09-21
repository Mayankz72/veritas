"""Ingest the built-in demo paper (Attention Is All You Need) from the local
PDF in ml-service/examples/, without needing the API server running.

Usage (from ml-service/, with the venv activated):
    python -m scripts.ingest_demo
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from app.models import Chunk, Document
from app.services.pdf_extraction import extract_document

DEMO_PDF = Path(__file__).resolve().parent.parent / "examples" / "attention-is-all-you-need.pdf"


def main() -> None:
    if not DEMO_PDF.exists():
        raise SystemExit(f"Demo PDF not found at {DEMO_PDF}")

    init_db()
    pdf_bytes = DEMO_PDF.read_bytes()
    title, page_count, chunks = extract_document(pdf_bytes)

    db = SessionLocal()
    try:
        document = Document(
            id=uuid.uuid4().hex, source_type="upload", title=title, page_count=page_count
        )
        db.add(document)
        for index, chunk in enumerate(chunks):
            db.add(
                Chunk(
                    id=uuid.uuid4().hex,
                    document_id=document.id,
                    order_index=index,
                    page=chunk.page,
                    section=chunk.section,
                    text=chunk.text,
                    bbox=list(chunk.bbox) if chunk.bbox else None,
                )
            )
        db.commit()

        print(f"Ingested '{title}' ({page_count} pages, {len(chunks)} chunks)")
        print(f"document id: {document.id}")
        print()
        print("First 5 sections seen:")
        seen: list[str | None] = []
        for c in chunks:
            if c.section not in seen:
                seen.append(c.section)
            if len(seen) >= 5:
                break
        for s in seen:
            print(f"  - {s}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
