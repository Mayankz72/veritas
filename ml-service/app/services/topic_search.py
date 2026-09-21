"""Topic -> candidate papers.

arXiv's own relevance ranking is term-frequency based, so a query like
"efficient attention" returns papers that repeat the words rather than the
ones that are *about* the idea, and it buries the newest work. So we search
two ways - by relevance, and newest-first within the last couple of years -
merge the two pools, and rerank by embedding similarity between the topic and
each paper's title + abstract (the same bge model used for retrieval). Recent
papers get a guaranteed share of the final list, as long as they are relevant.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from app.services.arxiv import ArxivPaper, search_arxiv
from app.services.embeddings import embed_query, embed_texts

OVERFETCH_FACTOR = 3
RECENT_YEARS = 2
# A recent paper is only pulled into the list if its relevance is within this
# much of the best match - "new" must not mean "off-topic".
RECENT_RELEVANCE_SLACK = 0.10


@dataclass
class Candidate:
    paper: ArxivPaper
    score: float  # cosine similarity between the topic and the paper's title + abstract
    is_recent: bool = False  # published within the last RECENT_YEARS years


def rank_by_topic(topic: str, papers: list[ArxivPaper]) -> list[Candidate]:
    if not papers:
        return []
    topic_vector = np.array(embed_query(topic))
    paper_vectors = np.array(embed_texts([f"{p.title}. {p.abstract}" for p in papers]))
    norms = np.linalg.norm(paper_vectors, axis=1) * np.linalg.norm(topic_vector)
    scores = (paper_vectors @ topic_vector) / np.where(norms == 0, 1, norms)
    ranked = sorted(zip(papers, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [Candidate(paper=paper, score=float(score)) for paper, score in ranked]


def pick_with_recent_share(
    ranked: list[Candidate], limit: int, recent_since: date
) -> list[Candidate]:
    """Best `limit` candidates by relevance, but with at least a third of them
    recent (when enough relevant recent papers exist)."""
    if not ranked:
        return []
    cutoff = recent_since.isoformat()
    for candidate in ranked:
        candidate.is_recent = candidate.paper.published >= cutoff

    best = ranked[0].score
    quota = max(1, limit // 3)
    chosen = [c for c in ranked if c.is_recent and c.score >= best - RECENT_RELEVANCE_SLACK][
        :quota
    ]
    for candidate in ranked:
        if len(chosen) == limit:
            break
        if candidate not in chosen:
            chosen.append(candidate)
    return sorted(chosen, key=lambda c: c.score, reverse=True)


def find_candidates(topic: str, limit: int = 5) -> list[Candidate]:
    recent_since = date.today() - timedelta(days=365 * RECENT_YEARS)
    pool = {
        p.arxiv_id: p for p in search_arxiv(topic, max_results=max(limit * OVERFETCH_FACTOR, 12))
    }
    try:
        newest = search_arxiv(
            topic,
            max_results=max(limit * 2, 10),
            sort_by="submittedDate",
            submitted_after=recent_since,
        )
    except Exception:  # noqa: BLE001 - recency is a bonus; a hiccup here must not fail the search
        newest = []
    for paper in newest:
        pool.setdefault(paper.arxiv_id, paper)
    return pick_with_recent_share(rank_by_topic(topic, list(pool.values())), limit, recent_since)
