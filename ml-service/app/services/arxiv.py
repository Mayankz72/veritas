import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date

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


# ---------------------------------------------------------------------------
# Topic search (arXiv's public Atom API - no key needed)
# ---------------------------------------------------------------------------

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
USER_AGENT = "veritas-paper-studio/0.1 (research tool; contact via GitHub)"
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


@dataclass
class ArxivPaper:
    arxiv_id: str  # without version suffix
    title: str
    authors: list[str]
    published: str  # ISO date, e.g. "2017-06-12"
    abstract: str


def build_search_query(topic: str, *, match_all: bool = True) -> str:
    """`efficient attention` -> `all:efficient AND all:attention`. arXiv's
    phrase search is too strict for free-text topics, so terms are combined
    with AND (falling back to OR when that returns nothing)."""
    words = _WORD_RE.findall(topic)[:8]
    joiner = " AND " if match_all else " OR "
    return joiner.join(f"all:{w}" for w in words)


def parse_atom_feed(xml_text: str) -> list[ArxivPaper]:
    root = ET.fromstring(xml_text)
    papers: list[ArxivPaper] = []
    for entry in root.findall("a:entry", ATOM_NS):
        raw_id = (entry.findtext("a:id", default="", namespaces=ATOM_NS) or "").strip()
        arxiv_id = _VERSION_SUFFIX_RE.sub("", raw_id.rsplit("/abs/", 1)[-1])
        if not ARXIV_ID_RE.match(arxiv_id):
            continue  # old-style ids (cs/0501001) - the ingestion path only handles new-style
        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=" ".join((entry.findtext("a:title", default="", namespaces=ATOM_NS)).split()),
                authors=[
                    " ".join((a.findtext("a:name", default="", namespaces=ATOM_NS)).split())
                    for a in entry.findall("a:author", ATOM_NS)
                ],
                published=(entry.findtext("a:published", default="", namespaces=ATOM_NS) or "")[
                    :10
                ],
                abstract=" ".join(
                    (entry.findtext("a:summary", default="", namespaces=ATOM_NS)).split()
                ),
            )
        )
    return papers


API_MIN_INTERVAL_SECONDS = 3.0  # arXiv asks for at most one API request every 3 seconds
_last_api_call = 0.0


def _throttle_api() -> None:
    global _last_api_call
    wait = API_MIN_INTERVAL_SECONDS - (time.monotonic() - _last_api_call)
    if wait > 0:
        time.sleep(wait)
    _last_api_call = time.monotonic()


def search_arxiv(
    topic: str,
    max_results: int = 15,
    *,
    sort_by: str = "relevance",
    submitted_after: date | None = None,
) -> list[ArxivPaper]:
    """`sort_by="submittedDate"` with `submitted_after` finds the newest papers on
    a topic, which arXiv's relevance ranking (keyword frequency) buries."""
    for match_all in (True, False):
        query = build_search_query(topic, match_all=match_all)
        if submitted_after is not None:
            window = f"{submitted_after:%Y%m%d}0000 TO {date.today():%Y%m%d}2359"
            query = f"({query}) AND submittedDate:[{window}]"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        _throttle_api()
        response = _get_with_retries(ARXIV_API_URL, params=params, timeout=30.0)
        papers = parse_atom_feed(response.text)
        if papers:
            return papers
    return []


def _get_with_retries(url: str, *, params: dict | None = None, timeout: float) -> httpx.Response:
    """arXiv occasionally answers 429/5xx or stalls; a couple of polite
    retries turns most of those into successes instead of a failed analysis."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = httpx.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
                timeout=timeout,
            )
            if response.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"{response.status_code} from {url}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_error = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in (
                429,
                500,
                502,
                503,
                504,
            ):
                raise
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"arXiv request failed after retries: {last_error}")


def download_arxiv_pdf(arxiv_id: str) -> bytes:
    """Sync counterpart of fetch_arxiv_pdf, for the background topic pipeline."""
    normalized = normalize_arxiv_id(arxiv_id)
    return _get_with_retries(f"https://arxiv.org/pdf/{normalized}", timeout=60.0).content
