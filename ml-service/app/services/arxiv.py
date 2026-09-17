import re

import httpx

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


class InvalidArxivId(ValueError):
    pass


def normalize_arxiv_id(raw: str) -> str:
    candidate = raw.strip()
    candidate = candidate.removeprefix("https://arxiv.org/abs/")
    candidate = candidate.removeprefix("https://arxiv.org/pdf/")
    candidate = candidate.removesuffix(".pdf")
    if not ARXIV_ID_RE.match(candidate):
        raise InvalidArxivId(f"'{raw}' is not a valid arXiv id (expected e.g. 1706.03762)")
    return candidate


async def fetch_arxiv_pdf(arxiv_id: str) -> bytes:
    normalized = normalize_arxiv_id(arxiv_id)
    url = f"https://arxiv.org/pdf/{normalized}"
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content
