from app.models import Chunk
from app.routers.documents import _document_embedding


def _chunk(section: str | None) -> Chunk:
    return Chunk(page=1, section=section, text="x")


def test_prefers_abstract_section_when_present():
    chunks = [_chunk(None), _chunk("Abstract"), _chunk("1 Introduction")]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]

    result = _document_embedding(chunks, embeddings)

    assert result == [0.0, 1.0]


def test_falls_back_to_first_three_chunks_without_an_abstract():
    chunks = [_chunk(None), _chunk("1 Introduction"), _chunk("2 Background"), _chunk("3 Method")]
    embeddings = [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [100.0, 0.0]]

    result = _document_embedding(chunks, embeddings)

    assert result == [3.0, 0.0]  # mean of the first 3, excludes the 4th


def test_returns_none_for_empty_document():
    assert _document_embedding([], []) is None


def test_case_insensitive_abstract_match():
    chunks = [_chunk("ABSTRACT"), _chunk("1 Introduction")]
    embeddings = [[2.0, 4.0], [0.0, 0.0]]

    result = _document_embedding(chunks, embeddings)

    assert result == [2.0, 4.0]
