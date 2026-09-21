import fitz
import pytest

from app.services.pdf_extraction import (
    CHUNK_TARGET_CHARS,
    _clean_text,
    extract_document,
)


def test_clean_text_strips_nul_bytes():
    # Postgres text columns reject NUL bytes outright; some real PDFs
    # (e.g. certain ligature encodings) produce them during extraction.
    assert _clean_text("Adam\x00 optimizer") == "Adam optimizer"
    assert "\x00" not in _clean_text("a\x00b\x00c")


def _build_pdf() -> bytes:
    """Synthetic 2-page PDF: a title, section headings (large/bold), and
    body paragraphs (small font) spaced far enough apart to land in
    separate text blocks, so we can exercise chunk-flush behaviour."""
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 80), "Test Paper Title", fontsize=24, fontname="hebo")
    page1.insert_text((72, 140), "1 Introduction", fontsize=16, fontname="hebo")

    body_para = "lorem ipsum dolor sit amet consectetur adipiscing elit " * 6  # ~330 chars
    y = 200
    # 4 paragraphs (~330 chars each, ~1320 total): the flush check compares
    # buffer length *before* adding the next block against CHUNK_TARGET_CHARS
    # (800), so paragraph 4 pushes the running buffer over the threshold and
    # forces a flush, producing 2 chunks rather than 1.
    for i in range(4):
        rc = page1.insert_textbox(
            fitz.Rect(72, y, 500, y + 60),
            f"PARA{i} {body_para}",
            fontsize=10,
            fontname="helv",
        )
        assert rc >= 0, f"paragraph {i} did not fit its textbox"
        y += 120  # large gap so PyMuPDF segments these as separate blocks

    page2 = doc.new_page()
    page2.insert_text((72, 80), "2 Method", fontsize=16, fontname="hebo")
    rc = page2.insert_textbox(
        fitz.Rect(72, 120, 500, 180),
        "PARA_PAGE2 " + body_para,
        fontsize=10,
        fontname="helv",
    )
    assert rc >= 0, "page 2 paragraph did not fit its textbox"

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture(scope="module")
def parsed():
    pdf_bytes = _build_pdf()
    title, page_count, chunks = extract_document(pdf_bytes)
    return title, page_count, chunks


def test_title_is_detected_not_a_heading(parsed):
    title, _, _ = parsed
    assert title == "Test Paper Title"


def test_page_count(parsed):
    _, page_count, _ = parsed
    assert page_count == 2


def test_headings_become_section_labels(parsed):
    _, _, chunks = parsed
    sections = {c.section for c in chunks}
    assert "1 Introduction" in sections
    assert "2 Method" in sections


def test_chunks_do_not_span_pages(parsed):
    _, _, chunks = parsed
    intro_markers = ("PARA0", "PARA1", "PARA2")
    page1_chunks = [c for c in chunks if any(m in c.text for m in intro_markers)]
    page2_chunks = [c for c in chunks if "PARA_PAGE2" in c.text]
    assert page1_chunks, "expected page-1 body chunks"
    assert page2_chunks, "expected page-2 body chunks"
    assert all(c.page == 1 for c in page1_chunks)
    assert all(c.page == 2 for c in page2_chunks)


def test_long_section_is_split_into_multiple_chunks(parsed):
    _, _, chunks = parsed
    intro_chunks = [c for c in chunks if c.section == "1 Introduction"]
    # 3 paragraphs of ~330 chars each (~990 total) with a flush target of
    # CHUNK_TARGET_CHARS should not all collapse into a single chunk.
    assert len(intro_chunks) >= 2
    for c in intro_chunks:
        assert len(c.text) <= CHUNK_TARGET_CHARS * 1.5


def test_chunks_have_bboxes(parsed):
    _, _, chunks = parsed
    body_chunks = [c for c in chunks if c.section is not None]
    assert body_chunks
    assert all(c.bbox is not None for c in body_chunks)
