import pytest

from app.services.arxiv import InvalidArxivId, normalize_arxiv_id


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1706.03762", "1706.03762"),
        ("1706.03762v7", "1706.03762v7"),
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
        ("  1706.03762  ", "1706.03762"),
    ],
)
def test_normalize_valid_ids(raw, expected):
    assert normalize_arxiv_id(raw) == expected


@pytest.mark.parametrize("raw", ["not-an-id", "17066.03762", "1706.037622", ""])
def test_normalize_rejects_invalid_ids(raw):
    with pytest.raises(InvalidArxivId):
        normalize_arxiv_id(raw)
