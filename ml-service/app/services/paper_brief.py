# ruff: noqa: E501  (long regexes and LLM prompt text read better unwrapped)
"""Structured per-paper brief: Problem / Method / Key results / Why it matters.

Every item is tied to a page and an exact quote from the paper, and every
item is checked by the grounding verifier - the same evidence-first loop as
the rest of the project, applied to a fixed four-part reading of a paper.

Two modes share one evidence-selection stage:

* extractive (no API key): the best real sentences from the paper are the
  brief. Nothing is generated, so nothing can be invented.
* LLM (LLM_PROVIDER configured): the model writes a plain-language statement
  from the selected evidence sentences and must cite which sentence supports
  it. The statement is then verified against the source passage, and numbers
  the model wrote that don't appear in that passage downgrade its label -
  a cheap, deterministic guard against the classic hallucinated-metric.
"""

from __future__ import annotations

import difflib
import json
import re
import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk as ChunkModel
from app.models import Claim as ClaimModel
from app.services.claim_generation import LLMProvider
from app.services.embeddings import embed_query, embed_texts
from app.services.grounding_verifier import GroundingVerifier
from app.services.text_utils import split_sentences

MAX_CANDIDATE_SENTENCES = 700
EVIDENCE_PER_ASPECT_FOR_LLM = 8
ITEMS_PER_ASPECT = 2


@dataclass(frozen=True)
class Aspect:
    key: str
    title: str
    query: str  # what the aspect is *about*, embedded to score sentences
    cues: re.Pattern
    sections: re.Pattern  # section headings where this kind of sentence usually lives
    position: float  # >0 prefers late in the paper, <0 prefers early, 0 neutral
    numeric_weight: float = 0.0  # reward sentences carrying numbers (results)
    llm_instruction: str = ""


ASPECTS: tuple[Aspect, ...] = (
    Aspect(
        key="problem",
        title="Problem",
        query="The problem this paper addresses: the limitation of existing approaches and the motivation for the work.",
        cues=re.compile(
            r"\b(problem|challeng\w*|limitation\w*|bottleneck\w*|difficult\w*|however|fail\w*|"
            r"suffer\w*|existing (?:methods|approaches|models|work)|lack\w*|inefficien\w*|"
            r"expensive|costly|motivat\w*|we address|aim to|remains? (?:a|an|unclear|open))\b",
            re.I,
        ),
        sections=re.compile(r"abstract|introduction|motivation|background", re.I),
        position=-1.0,
        llm_instruction="the problem the paper tackles and why existing approaches fall short",
    ),
    Aspect(
        key="method",
        title="Method",
        query="The method the authors propose: the model, architecture or technique and how it works.",
        cues=re.compile(
            r"\b(we (?:propose|introduce|present|develop|design|show that)|our (?:model|method|approach|"
            r"architecture|framework|algorithm)|is based (?:solely )?on|consists? of|composed of|"
            r"novel|dispens\w*|leverag\w*|employ\w*|uses?)\b",
            re.I,
        ),
        sections=re.compile(
            r"abstract|introduction|method|model|approach|architecture|framework", re.I
        ),
        position=-0.3,
        llm_instruction="what the authors propose and how it works, in concrete terms",
    ),
    Aspect(
        key="results",
        title="Key results",
        query="The key experimental results: measured performance, improvements over baselines and benchmark numbers.",
        cues=re.compile(
            r"\b(achiev\w*|outperform\w*|state[- ]of[- ]the[- ]art|improv\w*|surpass\w*|"
            r"BLEU|accuracy|F1|perplexity|score[sd]?|reduc\w*|faster|speed[- ]?up|results? show|"
            r"establish\w*|new (?:record|best)|"
            # outcome statements ("our results indicate", "we find", "consistently outperforms")
            r"results? (?:indicate|demonstrate|reveal|suggest)\w*|we (?:find|found|observe\w*)|"
            r"experiments? (?:show|demonstrate|reveal)\w*|consistently|more effectively|higher|lower)\b",
            re.I,
        ),
        sections=re.compile(r"abstract|result|experiment|evaluation|performance|conclusion", re.I),
        position=0.0,
        numeric_weight=0.05,
        llm_instruction="the headline measured results, with the concrete numbers the evidence gives",
    ),
    Aspect(
        key="matters",
        title="Why it matters",
        query="Why this work matters: its significance, implications, generality and future directions.",
        cues=re.compile(
            r"\b(enabl\w*|generali[sz]\w*|future|implication\w*|significan\w*|advanc\w*|allow\w*|"
            r"beyond|we believe|can be applied|open\w* up|impact\w*|promising|potential|"
            r"in (?:this )?(?:work|paper),? we)\b",
            re.I,
        ),
        sections=re.compile(r"conclusion|discussion|abstract|future|limitation", re.I),
        position=1.0,
        llm_instruction="why the work matters: what it enables, how general it is, what it opens up",
    ),
)
ASPECT_TITLES = tuple(a.title for a in ASPECTS)

_SKIP_SECTION_RE = re.compile(
    r"reference|bibliograph|acknowledg|appendix|appendices|supplement", re.I
)
_REFERENCE_LIKE_RE = re.compile(
    r"(arXiv preprint|Proceedings of|Journal of|Conference on|\bpp\.\s*\d|\bIn [A-Z][^.]{0,80}\b(19|20)\d{2}\b|"
    r"https?://|©|\bdoi\b)",
    re.I,
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass
class Sentence:
    text: str
    chunk_index: int
    chunk_id: str
    page: int
    section: str | None
    chunk_text: str


@dataclass
class BriefItem:
    aspect: Aspect
    text: str  # what the reader sees
    quote: str  # exact sentence from the paper it rests on
    sentence: Sentence
    similarity: float


def is_usable_sentence(text: str) -> bool:
    if not 45 <= len(text) <= 420:
        return False
    if len(text.split()) < 8:
        return False
    if not (text[0].isupper() or text[0].isdigit()):
        return False  # a fragment left over from a block boundary
    letters = sum(ch.isalpha() for ch in text)
    if letters / len(text) < 0.6:
        return False  # tables, equations, number soup
    if _REFERENCE_LIKE_RE.search(text):
        return False
    return not re.match(r"^(Figure|Fig\.|Table)\s+\d", text)


_ABSTRACT_MARKER_RE = re.compile(r"\b(?:Abstract|ABSTRACT)\b\s*[.:—–-]?\s*")
_KEYWORDS_RE = re.compile(r"\b(?:Keywords?|Index Terms)\b\s*[:—–-]", re.I)
_FRONT_MATTER_CHUNKS = 3
_FRONT_MATTER_WINDOW = 900  # chars: an author/affiliation block is never longer than this


def strip_front_matter(text: str) -> str:
    """Drops the author/affiliation block glued in front of the abstract, and
    the keywords line after it. Many PDFs extract as one block, so the first
    abstract sentence otherwise arrives as "Jane Doe [0000-...] Univ. of X ...
    Abstract. Retrieval-augmented generation has..." - a sentence that fails
    every usability check, taking the paper's opening problem statement with it."""
    marker = _ABSTRACT_MARKER_RE.search(text)
    if marker and marker.start() < _FRONT_MATTER_WINDOW:
        text = text[marker.end() :]
    keywords = _KEYWORDS_RE.search(text)
    if keywords and keywords.start() < _FRONT_MATTER_WINDOW:
        text = text[: keywords.start()]
    return text


def candidate_sentences(chunks: list[ChunkModel]) -> list[Sentence]:
    sentences: list[Sentence] = []
    in_back_matter = False
    for index, chunk in enumerate(chunks):
        if chunk.section and _SKIP_SECTION_RE.search(chunk.section):
            in_back_matter = True  # references/appendix run to the end of the paper
        if in_back_matter:
            continue
        body = strip_front_matter(chunk.text) if index < _FRONT_MATTER_CHUNKS else chunk.text
        for text in split_sentences(body):
            if is_usable_sentence(text):
                sentences.append(
                    Sentence(
                        text=text,
                        chunk_index=index,
                        chunk_id=chunk.id,
                        page=chunk.page,
                        section=chunk.section,
                        chunk_text=chunk.text,
                    )
                )
    return sentences


def _cosine(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    return (matrix @ vector) / np.where(norms == 0, 1.0, norms)


def score_sentences(
    aspect: Aspect, sentences: list[Sentence], similarities: np.ndarray, chunk_count: int
) -> np.ndarray:
    scores = np.zeros(len(sentences))
    for i, sentence in enumerate(sentences):
        score = float(similarities[i])
        score += 0.06 * min(len(aspect.cues.findall(sentence.text)), 3)
        if sentence.section and aspect.sections.search(sentence.section):
            score += 0.12
        if aspect.position and chunk_count > 1:
            relative_position = sentence.chunk_index / (chunk_count - 1)
            score += aspect.position * (relative_position - 0.5) * 0.1
        if aspect.numeric_weight:
            score += aspect.numeric_weight * min(len(_NUMBER_RE.findall(sentence.text)), 3)
        # A sentence needs enough body to stand alone ("The fundamental
        # constraint of sequential computation, however, remains." doesn't).
        if len(sentence.text) < 80:
            score -= 0.07
        elif len(sentence.text) > 300:
            score -= 0.05
        scores[i] = score
    return scores


def _near_duplicate(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.75


def select_top(
    sentences: list[Sentence],
    scores: np.ndarray,
    limit: int,
    exclude: set[str] | None = None,
) -> list[int]:
    chosen: list[int] = []
    for i in np.argsort(-scores):
        text = sentences[int(i)].text
        if exclude and text in exclude:
            continue
        if any(_near_duplicate(text, sentences[j].text) for j in chosen):
            continue
        chosen.append(int(i))
        if len(chosen) == limit:
            break
    return chosen


def _prune_candidates(sentences: list[Sentence]) -> list[Sentence]:
    """Bound embedding cost on very long papers: keep the sentences with the
    most cue-phrase evidence for any aspect."""
    if len(sentences) <= MAX_CANDIDATE_SENTENCES:
        return sentences
    prescore = [
        max(
            len(a.cues.findall(s.text)) + (2 if s.section and a.sections.search(s.section) else 0)
            for a in ASPECTS
        )
        for s in sentences
    ]
    keep = sorted(np.argsort(-np.array(prescore))[:MAX_CANDIDATE_SENTENCES])
    return [sentences[int(i)] for i in keep]


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

BRIEF_PROMPT = """You are writing a structured brief of a research paper for a reader who has not read it.

Below, for each aspect, are numbered evidence sentences quoted from the paper. For each aspect write 1 or 2 plain-language statements (each at most 40 words) that are FULLY supported by the evidence sentences listed under that aspect, and give the number of the single sentence that best supports each statement.

Rules:
- Use ONLY the evidence given. Do not add outside knowledge, background, or numbers that are not in the evidence.
- Write for a smart non-expert: say what the thing is and does, not just its name.
- Every published paper states its problem, method, results and significance, so do not leave an aspect empty: if the evidence for one is thin, write the closest statement the evidence does support, worded honestly (for example "The paper reports that ..."), and cite that sentence. Never invent a detail to fill the gap.

Paper title: {title}

{evidence_block}

Return JSON only, in exactly this shape:
{shape}
"""


def parse_json_object(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def is_verbatim(text: str, passage: str) -> bool:
    """`text` appears in `passage` once case, spacing, hyphenation and
    punctuation are ignored."""
    squashed = _squash(text)
    return len(squashed) >= 20 and squashed in _squash(passage)


def numbers_supported(statement: str, evidence_text: str) -> bool:
    """Every number the model wrote must literally appear in the source
    passage. Numbers are where LLM summaries hallucinate, and this is cheap
    to check exactly."""
    evidence_numbers = set(_NUMBER_RE.findall(evidence_text))
    return all(number in evidence_numbers for number in _NUMBER_RE.findall(statement))


def _llm_items(
    provider: LLMProvider,
    title: str,
    per_aspect_evidence: dict[str, list[tuple[Sentence, float]]],
    aspects: tuple[Aspect, ...] = ASPECTS,
) -> list[BriefItem]:
    blocks = []
    for aspect in aspects:
        lines = "\n".join(
            f"  [{n}] {sentence.text}"
            for n, (sentence, _) in enumerate(per_aspect_evidence[aspect.key], start=1)
        )
        blocks.append(f'Aspect "{aspect.key}" - {aspect.llm_instruction}:\n{lines}')
    shape = (
        "{"
        + ", ".join(f'"{a.key}": [{{"statement": "...", "evidence": 1}}]' for a in aspects)
        + "}"
    )
    raw = provider.complete(
        BRIEF_PROMPT.format(title=title, evidence_block="\n\n".join(blocks), shape=shape),
        json_mode=True,
    )
    parsed = parse_json_object(raw)

    items: list[BriefItem] = []
    for aspect in aspects:
        evidence = per_aspect_evidence[aspect.key]
        for entry in (parsed.get(aspect.key) or [])[:ITEMS_PER_ASPECT]:
            statement = str(entry.get("statement", "")).strip()
            if not statement or not evidence:
                continue
            try:
                cited = evidence[int(entry.get("evidence")) - 1]
            except (TypeError, ValueError, IndexError):
                # The model cited a sentence that doesn't exist: fall back to
                # the closest one instead of trusting a bad pointer.
                cited = max(
                    evidence,
                    key=lambda pair: difflib.SequenceMatcher(
                        None, statement.lower(), pair[0].text.lower()
                    ).ratio(),
                )
            items.append(
                BriefItem(
                    aspect=aspect,
                    text=statement,
                    quote=cited[0].text,
                    sentence=cited[0],
                    similarity=cited[1],
                )
            )
    return items


JUDGE_PROMPT = """You are a strict fact-checker. For each numbered item, decide whether the SOURCE passage supports the STATEMENT.

- "supported": every fact in the statement is stated in, or directly entailed by, the source.
- "partial": the source supports part of the statement, but the statement adds something the source does not say.
- "unsupported": the source does not support the statement.

Judge only against the source text, not against what you know about the paper.

{items_block}

Return JSON only: {{"items": [{{"id": 1, "label": "supported"}}, ...]}}
"""


def judge_statements(provider: LLMProvider, pairs: list[tuple[str, str]]) -> list[str | None]:
    """Entailment check of (statement, source passage) pairs by the LLM, in a
    single batched call. Returns a label per pair, or None where the judge
    gave no usable answer."""
    if not pairs:
        return []
    items_block = "\n\n".join(
        f"Item {n}\nSTATEMENT: {statement}\nSOURCE: {passage}"
        for n, (statement, passage) in enumerate(pairs, start=1)
    )
    parsed = parse_json_object(
        provider.complete(JUDGE_PROMPT.format(items_block=items_block), json_mode=True)
    )
    labels: list[str | None] = [None] * len(pairs)
    for entry in parsed.get("items", []):
        try:
            index = int(entry["id"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        label = str(entry.get("label", "")).lower()
        if 0 <= index < len(pairs) and label in ("supported", "partial", "unsupported"):
            labels[index] = label
    return labels


def final_label(
    judge_label: str | None,
    classifier_label: str | None,
    on_source: float | None,
    numbers_ok: bool,
) -> str | None:
    """Combines the LLM judge with the trained classifier and the number
    guard. Only ever lowers what the judge said, never raises it: the judge
    can be talked into agreeing with a fluent paraphrase, so a claim needs
    the guards' agreement to stay "supported"."""
    if judge_label is None:  # no judge (extractive mode, or the call failed)
        if classifier_label == "supported" and not numbers_ok:
            return "partial"
        return classifier_label
    label = judge_label
    if label == "supported" and (not numbers_ok or (on_source is not None and on_source < 0.5)):
        label = "partial"
    return label


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


ABSTRACT_MATCH_THRESHOLD = 0.9  # cosine; a near-copy of an abstract sentence scores ~0.95+
ABSTRACT_BONUS = 0.20
ABSTRACT_EVIDENCE_SLOTS = 4  # of the evidence sentences offered per aspect
BROAD_EVIDENCE_PER_ASPECT = 14  # for the repair pass on an aspect the first pass left empty
MIN_ABSTRACT_SENTENCE_CHARS = 40


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def _match_abstract(
    sentence_vectors: np.ndarray, abstract: str | None
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """A paper's abstract already states its problem, method, results and
    significance - the best evidence there is. We have the abstract from arXiv.

    Returns (1) a boost for PDF sentences that are near-copies of an abstract
    sentence - same content, but with a real page number and exact quote to
    cite - and (2) the abstract sentences that have *no* near-copy in the PDF
    text (mangled by two-column layout, glued to the author block...), so they
    can still be offered as evidence instead of being lost."""
    count = len(sentence_vectors)
    bonus = np.zeros(count)
    abstract_sentences = [
        s for s in split_sentences(abstract or "") if len(s) >= MIN_ABSTRACT_SENTENCE_CHARS
    ]
    if not abstract_sentences:
        return bonus, []
    abstract_vectors = np.array(embed_texts(abstract_sentences))
    if not count:
        return bonus, list(zip(abstract_sentences, abstract_vectors, strict=True))

    similarity = _unit_rows(sentence_vectors) @ _unit_rows(abstract_vectors).T
    bonus[similarity.max(axis=1) >= ABSTRACT_MATCH_THRESHOLD] = ABSTRACT_BONUS
    unmatched = [
        (text, abstract_vectors[j])
        for j, text in enumerate(abstract_sentences)
        if similarity[:, j].max() < ABSTRACT_MATCH_THRESHOLD
    ]
    return bonus, unmatched


def abstract_bonus(sentence_vectors: np.ndarray, abstract: str | None) -> np.ndarray:
    return _match_abstract(sentence_vectors, abstract)[0]


@dataclass
class Evidence:
    offered: dict[str, list[tuple[Sentence, float]]]  # tight set shown to the LLM
    broad: dict[str, list[tuple[Sentence, float]]]  # wider set for the repair pass
    extractive: dict[str, list[BriefItem]]  # verbatim fallback, always filled if any text exists


def select_evidence(chunks: list[ChunkModel], abstract: str | None = None) -> Evidence:
    """Scores every usable sentence against every aspect."""
    sentences = _prune_candidates(candidate_sentences(chunks))
    vectors = (
        np.array(embed_texts([s.text for s in sentences]))
        if sentences
        else np.zeros((0, len(embed_query("x"))))
    )
    from_abstract, unmatched = _match_abstract(vectors, abstract)

    if unmatched and chunks:
        # Attach each stray abstract sentence to the front-matter chunk it most
        # resembles, so it still has a page and a passage to be verified against.
        head = chunks[:_FRONT_MATTER_CHUNKS]
        head_vectors = np.array(embed_texts([c.text for c in head]))
        rescued_vectors = []
        for text, vector in unmatched:
            best = int(np.argmax(_cosine(head_vectors, vector)))
            # Only rescue a sentence that really is in the PDF text (ignoring the
            # hyphenation/spacing damage that made it miss the near-copy match). The
            # arXiv abstract can differ from the PDF's wording (other version, other
            # phrasing), and citing a page for text that isn't on it would be a false citation.
            if not is_verbatim(text, head[best].text):
                continue
            sentences.append(
                Sentence(
                    text=text,
                    chunk_index=best,
                    chunk_id=head[best].id,
                    page=head[best].page,
                    section="Abstract",
                    chunk_text=head[best].text,
                )
            )
            rescued_vectors.append(vector)
        if rescued_vectors:
            vectors = np.vstack([vectors, np.array(rescued_vectors)])
            from_abstract = np.concatenate(
                [from_abstract, np.full(len(rescued_vectors), ABSTRACT_BONUS)]
            )

    empty = {a.key: [] for a in ASPECTS}
    if not sentences:
        return Evidence(offered=dict(empty), broad=dict(empty), extractive=dict(empty))

    offered: dict[str, list[tuple[Sentence, float]]] = {}
    broad: dict[str, list[tuple[Sentence, float]]] = {}
    extractive: dict[str, list[BriefItem]] = {}
    used: set[str] = set()

    for aspect in ASPECTS:
        similarities = _cosine(vectors, np.array(embed_query(aspect.query)))
        scores = score_sentences(aspect, sentences, similarities, len(chunks)) + from_abstract

        top = select_top(sentences, scores, EVIDENCE_PER_ASPECT_FOR_LLM)
        abstract_only = np.where(from_abstract > 0, scores, -np.inf)
        reserved = [
            i
            for i in select_top(sentences, abstract_only, ABSTRACT_EVIDENCE_SLOTS)
            if from_abstract[i] > 0
        ]
        # Reserve slots for the abstract's own best sentences on this aspect, so e.g.
        # metric definitions from the experiments section can't crowd out the
        # abstract's statement of the findings.
        top = (reserved + [i for i in top if i not in reserved])[:EVIDENCE_PER_ASPECT_FOR_LLM]
        offered[aspect.key] = [(sentences[i], float(similarities[i])) for i in top]

        wide = select_top(sentences, scores, BROAD_EVIDENCE_PER_ASPECT)
        every_abstract = [int(i) for i in np.argsort(-abstract_only) if from_abstract[i] > 0]
        wide = list(dict.fromkeys(every_abstract + wide))
        broad[aspect.key] = [(sentences[i], float(similarities[i])) for i in wide]

        # The extractive brief avoids reusing a sentence across aspects.
        chosen = select_top(sentences, scores, ITEMS_PER_ASPECT, exclude=used)
        if not chosen:
            # A short paper can run out of sentences; better to repeat the best
            # one under a second heading than to leave an aspect empty.
            chosen = select_top(sentences, scores, 1)
        extractive[aspect.key] = [
            BriefItem(
                aspect=aspect,
                text=sentences[i].text,
                quote=sentences[i].text,
                sentence=sentences[i],
                similarity=float(similarities[i]),
            )
            for i in chosen
        ]
        used.update(sentences[i].text for i in chosen)
    return Evidence(offered=offered, broad=broad, extractive=extractive)


def assemble_items(
    provider: LLMProvider, title: str, evidence: Evidence
) -> tuple[list[BriefItem], str]:
    """Builds the brief so that no aspect is ever left empty, in three layers:

    1. the LLM writes each aspect from the tight evidence;
    2. any aspect it left empty is retried with much broader evidence (every
       abstract sentence plus the aspect's top sentences from the whole paper);
    3. anything still empty falls back to the paper's own best sentence for that
       aspect, quoted verbatim (which the UI marks as a direct quote).
    """
    by_aspect: dict[str, list[BriefItem]] = {a.key: [] for a in ASPECTS}
    mode = "extractive"

    if provider.supports_completion and any(evidence.offered.values()):
        try:
            for item in _llm_items(provider, title, evidence.offered):
                by_aspect[item.aspect.key].append(item)
            if any(by_aspect.values()):
                mode = f"llm:{provider.name}"
        except Exception:  # noqa: BLE001 - a failed LLM call must not sink the paper
            by_aspect = {a.key: [] for a in ASPECTS}

        missing = tuple(a for a in ASPECTS if not by_aspect[a.key] and evidence.broad[a.key])
        if missing and mode.startswith("llm:"):
            try:
                for item in _llm_items(provider, title, evidence.broad, aspects=missing):
                    by_aspect[item.aspect.key].append(item)
            except Exception:  # noqa: BLE001 - the verbatim fallback below still applies
                pass

    for aspect in ASPECTS:
        if not by_aspect[aspect.key]:
            fallback = evidence.extractive[aspect.key]
            by_aspect[aspect.key] = fallback if mode == "extractive" else fallback[:1]

    return [item for aspect in ASPECTS for item in by_aspect[aspect.key]], mode


def build_brief(
    db: Session,
    document_id: str,
    title: str,
    provider: LLMProvider,
    verifier: GroundingVerifier | None,
    abstract: str | None = None,
) -> tuple[list[ClaimModel], str]:
    """Builds and persists the four-part brief for one ingested document as
    ordinary Claim rows (section = "Problem" / "Method" / ...), so it shows up
    in the document workspace, health panel and exports like any other claim.
    Returns the claims and the mode that produced them."""
    chunks = list(
        db.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.order_index)
        ).scalars()
    )
    items, mode = assemble_items(provider, title, select_evidence(chunks, abstract))

    previous = db.execute(
        select(ClaimModel).where(
            ClaimModel.document_id == document_id,
            ClaimModel.section.in_(ASPECT_TITLES),
            ClaimModel.is_current.is_(True),
        )
    ).scalars()
    for claim in previous:
        claim.is_current = False

    llm_mode = mode.startswith("llm:")
    judge_labels: list[str | None] = [None] * len(items)
    if llm_mode:
        try:
            judge_labels = judge_statements(
                provider, [(item.text, item.sentence.chunk_text) for item in items]
            )
        except Exception:  # noqa: BLE001 - fall back to the classifier alone
            judge_labels = [None] * len(items)

    claim_models: list[ClaimModel] = []
    for item, judge_label in zip(items, judge_labels, strict=True):
        label, score = None, None
        if item.text == item.quote and is_verbatim(item.text, item.sentence.chunk_text):
            # A direct quote is supported by its own source by definition. Checked
            # with spacing/hyphenation/punctuation ignored, because PDF extraction
            # breaks words ("trans- formative") and the trained classifier - which
            # compares a sentence with a whole chunk - wrongly rejects such quotes.
            label, score = "supported", 1.0
        elif verifier is not None:
            numbers_ok = numbers_supported(item.text, item.sentence.chunk_text)
            if llm_mode:
                # Paraphrase: compare against the cited sentence, and report the
                # classifier's on-source probability as the confidence.
                classifier_label, _ = verifier.predict(item.text, item.quote)
                score = verifier.on_source_probability(item.text, item.quote)
                label = final_label(judge_label, classifier_label, score, numbers_ok)
            else:
                label, score = verifier.predict(item.text, item.sentence.chunk_text)
        elif judge_label is not None:
            label = judge_label
        claim_models.append(
            ClaimModel(
                id=uuid.uuid4().hex,
                document_id=document_id,
                section=item.aspect.title,
                text=item.text,
                source_chunk_ids=[item.sentence.chunk_id],
                page=item.sentence.page,
                quote=item.quote,
                retrieval_score=item.similarity,
                grounding_label=label,
                grounding_score=score,
                version=1,
                is_current=True,
            )
        )
    db.add_all(claim_models)
    db.commit()
    return claim_models, mode
