"""Heuristic PDF -> ParsedDocument extraction.

Uses PyMuPDF to pull text blocks with font metadata, classifies large/bold
short blocks as section headings, and merges the remaining body blocks into
page-and-section-tagged chunks sized for retrieval.

This is a heuristic baseline (font-size relative to the document's median
body size). Phase 7 of the roadmap replaces it with a trained layout model
and reports accuracy against this baseline.
"""

import re
from dataclasses import dataclass
from statistics import median

import fitz  # PyMuPDF

HEADING_SIZE_RATIO = 1.15
HEADING_MAX_CHARS = 120
CHUNK_TARGET_CHARS = 800
CHUNK_MIN_FLUSH_CHARS = 200
BOLD_FLAG = 1 << 4


@dataclass
class RawBlock:
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float
    bold: bool


@dataclass
class ExtractedChunk:
    page: int
    section: str | None
    text: str
    bbox: tuple[float, float, float, float] | None


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _extract_blocks(doc: fitz.Document) -> list[RawBlock]:
    blocks: list[RawBlock] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            if block.get("type") != 0:  # not a text block (e.g. image)
                continue
            lines = block.get("lines", [])
            if not lines:
                continue

            texts: list[str] = []
            sizes: list[float] = []
            bold_votes = 0
            span_count = 0
            for line in lines:
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if not span_text.strip():
                        continue
                    texts.append(span_text)
                    sizes.append(span.get("size", 0.0))
                    span_count += 1
                    flags = span.get("flags", 0)
                    if flags & BOLD_FLAG or "bold" in span.get("font", "").lower():
                        bold_votes += 1

            text = _clean_text(" ".join(texts))
            if not text:
                continue

            blocks.append(
                RawBlock(
                    page=page_index + 1,
                    bbox=tuple(block["bbox"]),
                    text=text,
                    font_size=median(sizes) if sizes else 0.0,
                    bold=span_count > 0 and bold_votes / span_count > 0.5,
                )
            )
    return blocks


def _body_font_size(blocks: list[RawBlock]) -> float:
    sizes = [b.font_size for b in blocks if b.font_size > 0]
    return median(sizes) if sizes else 10.0


def _is_heading(block: RawBlock, body_size: float) -> bool:
    if len(block.text) > HEADING_MAX_CHARS:
        return False
    if block.font_size >= body_size * HEADING_SIZE_RATIO:
        return True
    return block.bold and block.font_size >= body_size and len(block.text) < 80


def _union_bbox(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def _chunk_blocks(blocks: list[RawBlock], body_size: float) -> list[ExtractedChunk]:
    chunks: list[ExtractedChunk] = []
    current_section: str | None = None
    buffer_texts: list[str] = []
    buffer_boxes: list[tuple[float, float, float, float]] = []
    buffer_page: int | None = None

    def flush() -> None:
        nonlocal buffer_texts, buffer_boxes, buffer_page
        if buffer_texts and buffer_page is not None:
            chunks.append(
                ExtractedChunk(
                    page=buffer_page,
                    section=current_section,
                    text=" ".join(buffer_texts).strip(),
                    bbox=_union_bbox(buffer_boxes),
                )
            )
        buffer_texts = []
        buffer_boxes = []
        buffer_page = None

    for block in blocks:
        if _is_heading(block, body_size):
            flush()
            current_section = block.text
            continue

        buffer_len = sum(len(t) for t in buffer_texts)
        page_changed = buffer_page is not None and block.page != buffer_page
        would_overflow = buffer_len >= CHUNK_TARGET_CHARS

        if (page_changed and buffer_len >= CHUNK_MIN_FLUSH_CHARS) or would_overflow:
            flush()

        if buffer_page is None:
            buffer_page = block.page
        buffer_texts.append(block.text)
        buffer_boxes.append(block.bbox)

    flush()
    return chunks


ARXIV_HEADER_RE = re.compile(r"arxiv:|^\d{4}\.\d{4,5}", re.IGNORECASE)
MIN_TITLE_BLOCK_WIDTH = 60.0


def _guess_title(blocks: list[RawBlock], body_size: float) -> str | None:
    first_page_blocks = [b for b in blocks if b.page == 1]
    candidates = [
        b
        for b in first_page_blocks
        if b.font_size >= body_size * 1.3
        and (b.bbox[2] - b.bbox[0]) >= MIN_TITLE_BLOCK_WIDTH
        and not ARXIV_HEADER_RE.search(b.text)
    ]
    if not candidates:
        return None
    # Topmost (smallest y0) qualifying block is the title, not the largest font
    # (running headers/watermarks can be larger than the actual title text).
    return min(candidates, key=lambda b: b.bbox[1]).text


def extract_document(
    pdf_bytes: bytes, title_hint: str | None = None
) -> tuple[str, int, list[ExtractedChunk]]:
    """Parse raw PDF bytes into a title, page count, and evidence chunks."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = len(doc)
        blocks = _extract_blocks(doc)
        body_size = _body_font_size(blocks)
        chunks = _chunk_blocks(blocks, body_size)
        title = title_hint or _guess_title(blocks, body_size) or "Untitled document"
        return title, page_count, chunks
    finally:
        doc.close()
