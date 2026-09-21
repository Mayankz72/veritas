# ruff: noqa: E501  (long regexes and LLM prompt text read better unwrapped)
"""How the papers in a topic fit together.

Relationships are computed from evidence in the papers themselves, not asked
of a model:

* "cites" - paper A's own text (its reference list) contains paper B's title,
  and B is older than A. The page and the surrounding snippet are kept as proof.
* "similar" - no citation link, but the abstracts sit close in embedding space
  and share distinctive terms (TF-IDF over the abstracts).

With an LLM configured, the *narrative* (overview / how they relate / reading
order) is written by the model from those facts plus the verified per-paper
briefs, and labels it cites are validated against the real paper list.
Without one, the narrative is assembled from templates over the same facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.services.claim_generation import LLMProvider
from app.services.paper_brief import parse_json_object

# Words that appear in nearly every abstract and say nothing about *which* idea.
_GENERIC_TERMS = {
    "paper",
    "propose",
    "proposed",
    "present",
    "presents",
    "show",
    "shows",
    "shown",
    "results",
    "result",
    "method",
    "methods",
    "approach",
    "approaches",
    "model",
    "models",
    "based",
    "using",
    "use",
    "used",
    "novel",
    "new",
    "work",
    "study",
    "task",
    "tasks",
    "performance",
    "existing",
    "state",
    "art",
    "experiments",
    "experimental",
    "demonstrate",
    "demonstrates",
    "address",
    "problem",
    "problems",
    "improve",
    "improves",
    "significantly",
    "compared",
    "outperforms",
}
_STOP_WORDS = list(ENGLISH_STOP_WORDS | _GENERIC_TERMS)
_TITLE_MIN_MATCH_CHARS = 14


@dataclass
class PaperInput:
    paper_id: str
    label: str  # "P1", "P2", ... - what the LLM and the templates call a paper
    title: str
    arxiv_id: str
    published: str  # ISO date
    abstract: str
    chunks: list[tuple[int, str]]  # (page, text) for the full paper
    embedding: list[float] | None
    brief: dict[str, list[str]] = field(default_factory=dict)  # aspect key -> statements

    @property
    def year(self) -> str:
        return self.published[:4]

    @property
    def short_title(self) -> str:
        head = re.split(r"[:—–]| - ", self.title, maxsplit=1)[0].strip()
        return head if len(head) >= 4 else self.title


@dataclass
class Relationship:
    from_paper_id: str
    to_paper_id: str
    kind: str  # "cites" | "similar"
    explanation: str
    page: int | None = None
    snippet: str | None = None
    shared_terms: list[str] = field(default_factory=list)
    similarity: float | None = None


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _title_variants(paper: PaperInput) -> list[str]:
    """Full title, plus the part before a subtitle colon (references often
    abbreviate). Each must be long enough not to match by accident."""
    variants = [paper.title]
    head = re.split(r":", paper.title, maxsplit=1)[0]
    if head != paper.title:
        variants.append(head)
    return [v for v in variants if len(_normalize(v)) >= _TITLE_MIN_MATCH_CHARS]


def _normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalized text plus, for each normalized character, its index in the
    original - so a match found in normalized space (immune to hyphenation and
    spacing quirks from PDF extraction, like "Pre- training") can be shown in
    its original surroundings."""
    characters: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(text.lower()):
        if character.isascii() and character.isalnum():
            characters.append(character)
            offsets.append(index)
    return "".join(characters), offsets


def find_title_mention(citing: PaperInput, cited: PaperInput) -> tuple[int, str] | None:
    """Where (page, snippet) `citing`'s text mentions `cited`'s title, if anywhere."""
    for variant in _title_variants(cited):
        target = _normalize(variant)
        for page, text in citing.chunks:
            normalized, offsets = _normalize_with_offsets(text)
            found = normalized.find(target)
            if found == -1:
                continue
            match_start = offsets[found]
            match_end = offsets[found + len(target) - 1] + 1
            start, end = max(0, match_start - 90), min(len(text), match_end + 50)
            snippet = text[start:end].strip()
            return page, ("..." if start else "") + snippet + ("..." if end < len(text) else "")
    return None


def detect_citations(papers: list[PaperInput]) -> list[Relationship]:
    edges: list[Relationship] = []
    for citing in papers:
        for cited in papers:
            if citing.paper_id == cited.paper_id or not (cited.published < citing.published):
                continue  # a paper can't cite one that came after it
            mention = find_title_mention(citing, cited)
            if mention:
                page, snippet = mention
                edges.append(
                    Relationship(
                        from_paper_id=citing.paper_id,
                        to_paper_id=cited.paper_id,
                        kind="cites",
                        explanation=(
                            f"{citing.short_title} cites {cited.short_title} "
                            f"(its title appears in the reference list, p.{page})."
                        ),
                        page=page,
                        snippet=snippet,
                    )
                )
    return edges


def distinctive_terms(papers: list[PaperInput], top_n: int = 4) -> dict[tuple[int, int], list[str]]:
    """Terms two abstracts share, ranked by how weakly the *weaker* of the
    two weights them (TF-IDF), so a term must matter to both papers."""
    if len(papers) < 2:
        return {}
    try:
        vectorizer = TfidfVectorizer(
            stop_words=_STOP_WORDS, ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        matrix = vectorizer.fit_transform([p.abstract or p.title for p in papers]).toarray()
    except ValueError:  # empty vocabulary
        return {}
    vocabulary = np.array(vectorizer.get_feature_names_out())
    shared: dict[tuple[int, int], list[str]] = {}
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            weight = np.minimum(matrix[i], matrix[j])
            terms: list[str] = []
            for index in np.argsort(-weight):
                if weight[index] <= 0 or len(terms) == top_n:
                    break
                term = str(vocabulary[index])
                # Skip a unigram already covered by a chosen bigram ("attention" in "self attention").
                if any(term in chosen.split() or chosen in term.split() for chosen in terms):
                    continue
                terms.append(term)
            shared[(i, j)] = terms
    return shared


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denominator = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / denominator) if denominator else 0.0


def similarity_relationships(
    papers: list[PaperInput], citations: list[Relationship]
) -> list[Relationship]:
    """Links every paper that has no citation tie to the paper closest to it
    in embedding space, so no paper is left floating with no stated relation."""
    connected = {e.from_paper_id for e in citations} | {e.to_paper_id for e in citations}
    terms = distinctive_terms(papers)
    index_of = {p.paper_id: i for i, p in enumerate(papers)}
    edges: list[Relationship] = []
    seen: set[frozenset[str]] = set()

    for paper in papers:
        if paper.paper_id in connected or paper.embedding is None:
            continue
        candidates = [
            (other, _cosine(paper.embedding, other.embedding))
            for other in papers
            if other.paper_id != paper.paper_id and other.embedding is not None
        ]
        if not candidates:
            continue
        other, similarity = max(candidates, key=lambda pair: pair[1])
        pair_key = frozenset((paper.paper_id, other.paper_id))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        i, j = sorted((index_of[paper.paper_id], index_of[other.paper_id]))
        shared = terms.get((i, j), [])
        first, second = sorted((paper, other), key=lambda p: p.published)
        edges.append(
            Relationship(
                from_paper_id=second.paper_id,
                to_paper_id=first.paper_id,
                kind="similar",
                explanation=(
                    f"{second.short_title} and {first.short_title} address related ground"
                    + (f" (shared terms: {', '.join(shared)})" if shared else "")
                    + " but neither cites the other in the text we read."
                ),
                shared_terms=shared,
                similarity=similarity,
            )
        )
    return edges


def _first(paper: PaperInput, aspect: str) -> str:
    statements = paper.brief.get(aspect) or []
    return statements[0] if statements else ""


def comparison_rows(papers: list[PaperInput]) -> list[dict]:
    return [
        {
            "paper_id": p.paper_id,
            "label": p.label,
            "title": p.title,
            "year": p.year,
            "problem": _first(p, "problem"),
            "method": _first(p, "method"),
            "results": _first(p, "results"),
            "matters": _first(p, "matters"),
        }
        for p in sorted(papers, key=lambda p: p.published)
    ]


def template_narrative(
    topic: str, papers: list[PaperInput], relationships: list[Relationship]
) -> dict:
    ordered = sorted(papers, key=lambda p: p.published)
    by_id = {p.paper_id: p for p in papers}
    cites = [r for r in relationships if r.kind == "cites"]

    span = (
        f"{ordered[0].year}–{ordered[-1].year}"
        if ordered[0].year != ordered[-1].year
        else ordered[0].year
    )
    overview = f'{len(papers)} papers on "{topic}" ({span}).'
    if cites:
        overview += f" {len(cites)} of them cite one another, which gives the reading order below."
    else:
        overview += (
            " None of them cite one another in the text we read, so they are independent "
            "takes on related ground rather than a single lineage."
        )

    how_they_fit: list[dict] = []
    for relationship in relationships:
        citing, cited = by_id[relationship.from_paper_id], by_id[relationship.to_paper_id]
        if relationship.kind == "cites":
            text = f"{citing.short_title} builds on {cited.short_title}."
            method = _first(citing, "method")
            if method:
                text += f" {method}"
        else:
            text = relationship.explanation
        how_they_fit.append({"paper_ids": [citing.paper_id, cited.paper_id], "text": text})

    return {
        "mode": "template",
        "overview": overview,
        "how_they_fit": how_they_fit,
        "reading_order": [p.paper_id for p in ordered],
    }


NARRATIVE_PROMPT = """You are writing the "how these papers fit together" section of a literature brief on the topic "{topic}".

Papers (label, title, year), each with a brief already checked against the paper's text:
{papers_block}

Facts established from the papers' own text:
{facts_block}

Write:
1. "overview": 3-4 sentences on the big picture: what problem this line of work is about and how the approaches relate or differ.
2. "how_they_fit": 2 to 5 items. Each names the papers involved (by label) and states in at most 40 words how they relate (builds on, is an alternative to, extends, fixes a limitation of...).
3. "reading_order": every label, in the best order to read them.

Rules: base every relation ONLY on the briefs and facts above. If two papers have no evident relationship, say they are independent approaches to the same problem instead of inventing a link. Refer to papers by their label (P1, P2...).

Return JSON only: {{"overview": "...", "how_they_fit": [{{"papers": ["P1", "P2"], "text": "..."}}], "reading_order": ["P1", "P2"]}}
"""


def _humanize(text: str, papers: list[PaperInput]) -> str:
    """Swap the labels the model was given (P1, P2...) for readable names."""
    by_label = {p.label: f"{p.short_title} ({p.year})" for p in papers}
    return re.sub(r"\bP\d+\b", lambda m: by_label.get(m.group(0), m.group(0)), text)


def llm_narrative(
    provider: LLMProvider,
    topic: str,
    papers: list[PaperInput],
    relationships: list[Relationship],
) -> dict:
    by_id = {p.paper_id: p for p in papers}
    ordered = sorted(papers, key=lambda p: p.published)

    papers_block = "\n\n".join(
        f"{p.label} - {p.title} ({p.year})\n"
        + "\n".join(
            f"  {name}: {_first(p, key)}"
            for key, name in (
                ("problem", "Problem"),
                ("method", "Method"),
                ("results", "Key results"),
                ("matters", "Why it matters"),
            )
            if _first(p, key)
        )
        for p in ordered
    )
    facts = []
    for r in relationships:
        a, b = by_id[r.from_paper_id], by_id[r.to_paper_id]
        if r.kind == "cites":
            facts.append(f"- {a.label} cites {b.label} (found in {a.label}'s reference list).")
        else:
            terms = (
                f" Shared distinctive terms: {', '.join(r.shared_terms)}." if r.shared_terms else ""
            )
            facts.append(
                f"- {a.label} and {b.label} are close in topic but do not cite each other.{terms}"
            )
    if not facts:
        facts.append("- No citation links between these papers were found in their text.")

    raw = provider.complete(
        NARRATIVE_PROMPT.format(
            topic=topic, papers_block=papers_block, facts_block="\n".join(facts)
        ),
        json_mode=True,
    )
    parsed = parse_json_object(raw)

    label_to_id = {p.label: p.paper_id for p in papers}
    how_they_fit = []
    for entry in parsed.get("how_they_fit", []):
        ids = [label_to_id[name] for name in entry.get("papers", []) if name in label_to_id]
        text = _humanize(str(entry.get("text", "")).strip(), papers)
        if ids and text:
            how_they_fit.append({"paper_ids": ids, "text": text})

    reading_order = [
        label_to_id[name] for name in parsed.get("reading_order", []) if name in label_to_id
    ]
    reading_order += [
        p.paper_id for p in ordered if p.paper_id not in reading_order
    ]  # never drop a paper

    overview = _humanize(str(parsed.get("overview", "")).strip(), papers)
    if not overview:
        raise ValueError("model returned no overview")
    return {
        "mode": f"llm:{provider.name}",
        "overview": overview,
        "how_they_fit": how_they_fit,
        "reading_order": reading_order,
    }


def build_synthesis(
    topic: str, papers: list[PaperInput], provider: LLMProvider | None = None
) -> dict:
    ordered = sorted(papers, key=lambda p: p.published)
    citations = detect_citations(papers)
    relationships = citations + similarity_relationships(papers, citations)

    narrative = None
    if provider is not None and provider.supports_completion and len(papers) >= 2:
        try:
            narrative = llm_narrative(provider, topic, papers, relationships)
        except Exception:  # noqa: BLE001 - never lose the computed facts to a flaky model call
            narrative = None
    if narrative is None:
        narrative = template_narrative(topic, papers, relationships)

    cited_ids = {r.to_paper_id for r in citations}
    return {
        **narrative,
        "timeline": [
            {
                "paper_id": p.paper_id,
                "label": p.label,
                "title": p.title,
                "year": p.year,
                "published": p.published,
            }
            for p in ordered
        ],
        "relationships": [r.__dict__ for r in relationships],
        "comparison": comparison_rows(papers),
        "foundational": [p.paper_id for p in ordered if p.paper_id in cited_ids],
    }
