# ruff: noqa: E501  (test data strings read better unwrapped)
"""DB-free unit tests for the topic pipeline's pure logic: arXiv parsing,
sentence selection, the verification-label guards, citation detection and
the synthesis fallbacks."""

import json

import numpy as np
import pytest

from app.services.arxiv import build_search_query, parse_atom_feed
from app.services.claim_generation import ExtractiveProvider, LLMProvider
from app.services.paper_brief import (
    ASPECTS,
    Sentence,
    final_label,
    is_usable_sentence,
    judge_statements,
    numbers_supported,
    parse_json_object,
    score_sentences,
    select_top,
)
from app.services.text_utils import split_sentences
from app.services.topic_synthesis import (
    PaperInput,
    build_synthesis,
    detect_citations,
    find_title_mention,
    template_narrative,
)

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All
      You Need</title>
    <summary>  The dominant models   are recurrent.  </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/cs/0501001v1</id>
    <published>2005-01-01T00:00:00Z</published>
    <title>Old style id</title>
    <summary>skipped</summary>
  </entry>
</feed>"""


def test_parse_atom_feed_normalizes_and_skips_old_style_ids():
    papers = parse_atom_feed(ATOM)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "1706.03762"  # version suffix stripped
    assert paper.title == "Attention Is All You Need"  # whitespace collapsed
    assert paper.abstract == "The dominant models are recurrent."
    assert paper.published == "2017-06-12"
    assert paper.authors == ["Ashish Vaswani", "Noam Shazeer"]


def test_build_search_query_combines_terms():
    assert build_search_query("efficient attention!") == "all:efficient AND all:attention"
    assert build_search_query("efficient attention", match_all=False) == (
        "all:efficient OR all:attention"
    )


def test_split_sentences_keeps_abbreviations_together():
    text = "Vaswani et al. proposed it (see Fig. 3). It works. It scales, e.g. to long inputs."
    assert split_sentences(text) == [
        "Vaswani et al. proposed it (see Fig. 3).",
        "It works.",
        "It scales, e.g. to long inputs.",
    ]


@pytest.mark.parametrize(
    "text,usable",
    [
        ("We propose a new architecture that relies entirely on attention mechanisms.", True),
        ("Too short to be useful.", False),
        ("and then a fragment that starts lowercase but is otherwise long enough to pass", False),
        (
            "Figure 3: The Transformer model architecture with encoder and decoder stacks shown.",
            False,
        ),
        (
            "Vaswani, A. Attention is all you need. arXiv preprint arXiv:1706.03762, 2017 in press.",  # noqa: E501
            False,
        ),
        ("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29", False),
    ],
)
def test_is_usable_sentence(text, usable):
    assert is_usable_sentence(text) is usable


def _sentence(text: str, index: int = 0, section: str | None = None) -> Sentence:
    return Sentence(
        text=text, chunk_index=index, chunk_id=f"c{index}", page=1, section=section, chunk_text=text
    )


def test_aspect_scoring_prefers_the_right_kind_of_sentence():
    method = _sentence("We propose the Widget, a model that consists of stacked attention layers.")
    result = _sentence(
        "The Widget achieves 28.4 BLEU, outperforming the best reported results by 2.0."
    )
    plain = _sentence(
        "The committee met on Tuesday afternoon to discuss the catering arrangements."
    )
    sentences = [method, result, plain]
    flat = np.zeros(len(sentences))  # isolate the cue/number logic from embedding similarity

    by_key = {a.key: a for a in ASPECTS}
    method_scores = score_sentences(by_key["method"], sentences, flat, chunk_count=1)
    results_scores = score_sentences(by_key["results"], sentences, flat, chunk_count=1)
    assert select_top(sentences, method_scores, 1) == [0]
    assert select_top(sentences, results_scores, 1) == [1]


def test_select_top_skips_near_duplicates_and_excluded():
    a = _sentence("We propose the Widget model built from stacked attention layers.")
    b = _sentence("We propose the Widget model built from stacked attention layers!")
    c = _sentence("A completely different sentence about optimisation and learning rates here.")
    scores = np.array([3.0, 2.9, 1.0])
    assert select_top([a, b, c], scores, 2) == [0, 2]  # b is a near-duplicate of a
    assert select_top([a, b, c], scores, 2, exclude={a.text}) == [1, 2]


def test_numbers_supported_flags_invented_numbers():
    evidence = "The model reaches 28.4 BLEU on the English-to-German task."
    assert numbers_supported("It reaches 28.4 BLEU.", evidence)
    assert not numbers_supported("It reaches 29.9 BLEU.", evidence)
    assert numbers_supported("It reaches a new state of the art.", evidence)  # no numbers claimed


@pytest.mark.parametrize(
    "judge,classifier,on_source,numbers_ok,expected",
    [
        ("supported", "partial", 0.9, True, "supported"),
        ("supported", "unsupported", 0.2, True, "partial"),  # classifier disagrees -> downgrade
        ("supported", "partial", 0.9, False, "partial"),  # invented number -> downgrade
        ("partial", "supported", 0.99, True, "partial"),  # never raised above the judge
        ("unsupported", "supported", 0.99, True, "unsupported"),
        (None, "supported", 1.0, True, "supported"),  # no judge: classifier decides
        (None, "supported", 1.0, False, "partial"),
        (None, None, None, True, None),
    ],
)
def test_final_label_only_ever_lowers_the_judges_verdict(
    judge, classifier, on_source, numbers_ok, expected
):
    assert final_label(judge, classifier, on_source, numbers_ok) == expected


def test_parse_json_object_handles_code_fences_and_chatter():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here you go: {"a": {"b": 2}} Hope that helps.') == {
        "a": {"b": 2}
    }
    with pytest.raises(ValueError):
        parse_json_object("no json here")


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, reply: str | Exception):
        self.reply = reply

    def generate_claims(self, topic, passages, max_claims):
        return []

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_judge_statements_maps_labels_by_item_number():
    provider = FakeLLM(
        json.dumps(
            {
                "items": [
                    {"id": 2, "label": "Unsupported"},
                    {"id": 1, "label": "supported"},
                    {"id": 9, "label": "supported"},  # out of range: ignored
                    {"id": 3, "label": "bogus"},  # unknown label: ignored
                ]
            }
        )
    )
    labels = judge_statements(provider, [("s1", "p1"), ("s2", "p2"), ("s3", "p3")])
    assert labels == ["supported", "unsupported", None]


def _paper(pid, label, title, published, chunks=(), abstract="", brief=None):
    return PaperInput(
        paper_id=pid,
        label=label,
        title=title,
        arxiv_id="0000.00000",
        published=published,
        abstract=abstract,
        chunks=list(chunks),
        embedding=None,
        brief=brief or {},
    )


BERT = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"


def test_citation_found_despite_pdf_hyphenation_and_snippet_shows_the_match():
    reference_chunk = (
        "Yinhan Liu et al. 2019. Some other paper that is cited here first. arXiv preprint. "
        "Jacob Devlin, Ming-Wei Chang. 2018. BERT: Pre- training of deep bidirectional "
        "transformers for language under- standing. In NAACL. Next reference follows."
    )
    citing = _paper(
        "r", "P2", "RoBERTa", "2019-07-26", chunks=[(1, "intro"), (10, reference_chunk)]
    )
    cited = _paper("b", "P1", BERT, "2018-10-11")
    page, snippet = find_title_mention(citing, cited)
    assert page == 10
    assert "Pre- training" in snippet  # the snippet is centred on the real match
    assert "Some other paper" not in snippet.split("BERT")[1]


def test_a_paper_cannot_cite_a_later_one():
    older = _paper("a", "P1", "Attention Is All You Need", "2017-06-12", chunks=[(1, BERT)])
    newer = _paper("b", "P2", BERT, "2018-10-11", chunks=[(1, "Attention is all you need.")])
    edges = detect_citations([older, newer])
    assert [(e.from_paper_id, e.to_paper_id) for e in edges] == [("b", "a")]


def test_short_titles_do_not_create_false_citations():
    citing = _paper(
        "a", "P1", "Some Long Paper Title Here", "2020-01-01", chunks=[(1, "We use Adam.")]
    )
    cited = _paper("b", "P2", "Adam", "2014-12-22")
    assert detect_citations([citing, cited]) == []


def _two_papers():
    a = _paper(
        "a",
        "P1",
        "Attention Is All You Need",
        "2017-06-12",
        abstract="Self attention transformer sequence transduction.",
        brief={"problem": ["Recurrence is slow."], "method": ["Use attention only."]},
    )
    b = _paper(
        "b",
        "P2",
        BERT,
        "2018-10-11",
        chunks=[(9, "Vaswani. 2017. Attention is all you need. In NIPS.")],
        abstract="Bidirectional transformer pretraining for language understanding.",
        brief={"method": ["Pretrain bidirectionally."]},
    )
    return [a, b]


def test_template_narrative_states_when_papers_do_not_cite_each_other():
    papers = _two_papers()
    papers[1].chunks = []  # no citation evidence
    narrative = template_narrative("transformers", papers, [])
    assert "None of them cite one another" in narrative["overview"]
    assert narrative["reading_order"] == ["a", "b"]  # chronological


def test_build_synthesis_without_llm_uses_template_and_finds_citation():
    result = build_synthesis("transformers", _two_papers(), ExtractiveProvider())
    assert result["mode"] == "template"
    assert [(r["from_paper_id"], r["to_paper_id"]) for r in result["relationships"]] == [("b", "a")]
    assert result["foundational"] == ["a"]
    assert [row["paper_id"] for row in result["comparison"]] == ["a", "b"]


def test_build_synthesis_llm_narrative_validates_labels_and_keeps_every_paper():
    reply = json.dumps(
        {
            "overview": "P2 builds on P1.",
            "how_they_fit": [
                {"papers": ["P2", "P1"], "text": "P2 extends P1."},
                {"papers": ["P99"], "text": "ghost paper"},  # unknown label: dropped
            ],
            "reading_order": ["P2", "P77"],  # P77 unknown, P1 omitted -> appended
        }
    )
    result = build_synthesis("transformers", _two_papers(), FakeLLM(reply))
    assert result["mode"] == "llm:fake"
    assert result["overview"] == "BERT (2018) builds on Attention Is All You Need (2017)."
    assert len(result["how_they_fit"]) == 1
    assert result["how_they_fit"][0]["paper_ids"] == ["b", "a"]
    assert result["reading_order"] == ["b", "a"]


def test_build_synthesis_falls_back_to_template_when_the_llm_fails():
    result = build_synthesis("transformers", _two_papers(), FakeLLM(RuntimeError("503")))
    assert result["mode"] == "template"
    assert result["relationships"]  # the computed facts survive a failed model call


# ---- abstract-guided evidence -------------------------------------------------

_VOCAB = ["retrieval", "chunking", "results", "coherence", "cost", "optimizer", "committee", "tea"]


def _fake_embed(texts):
    """Bag-of-words vectors: deterministic, offline, and similar for similar text."""
    return [[float(word in text.lower()) for word in _VOCAB] + [0.01] for text in texts]


def test_abstract_bonus_flags_near_copies_of_abstract_sentences(monkeypatch):
    from app.services import paper_brief

    monkeypatch.setattr(paper_brief, "embed_texts", _fake_embed)
    pdf_sentences = [
        "Our results show retrieval keeps coherence at a higher cost for chunking.",
        "The committee drank tea and discussed the optimizer schedule at length today.",
    ]
    abstract = "Our results show retrieval keeps coherence at a higher cost for chunking."
    vectors = np.array(_fake_embed(pdf_sentences))
    bonus = paper_brief.abstract_bonus(vectors, abstract)
    assert bonus[0] == paper_brief.ABSTRACT_BONUS
    assert bonus[1] == 0.0
    assert not paper_brief.abstract_bonus(vectors, None).any()  # no abstract -> no boost


def test_abstract_findings_sentence_is_reserved_a_slot_even_when_outscored(monkeypatch):
    """Regression: metric-definition sentences (lots of cue words, high similarity to a
    "results" query) used to crowd the abstract's statement of the findings out of the
    evidence offered to the LLM."""
    from types import SimpleNamespace

    from app.services import paper_brief

    vocab = ["accuracy", "score", "results", "coherence"]

    def embed(texts):
        return [[float(word in text.lower()) for word in vocab] + [0.01] for text in texts]

    monkeypatch.setattr(paper_brief, "embed_texts", embed)
    # a "results" query that, like the real one, sits closest to metric talk
    monkeypatch.setattr(paper_brief, "embed_query", lambda text: [1.0, 1.0, 1.0, 0.0, 0.01])

    finding = (
        "Our results show contextual retrieval keeps coherence at a higher computational cost."
    )
    metric_definitions = [
        "The F1 score balances precision and recall to compare accuracy across all the evaluated systems.",
        "Accuracy is reported as the fraction of correct answers among every question in the benchmark.",
        "We compute the BLEU score on the held-out validation split for each individual model variant.",
        "Scores are averaged over five random seeds to reduce variance in the reported accuracy values.",
        "The accuracy metric follows the standard protocol described by the original benchmark authors.",
        "Every score in the table uses the same tokenizer and the identical evaluation script throughout.",
        "Faster evaluation uses a subset while the F1 score and accuracy remain comparable in practice.",
        "Higher scores indicate better accuracy under the metric definitions given in this section here.",
    ]
    chunks = [
        SimpleNamespace(id="a", page=1, section="Abstract", text=finding),
        SimpleNamespace(
            id="b", page=9, section="4 Experimental Setup", text=" ".join(metric_definitions)
        ),
    ]

    without = paper_brief.select_evidence(chunks, abstract=None).offered
    with_abstract = paper_brief.select_evidence(chunks, abstract=finding).offered

    assert finding not in [s.text for s, _ in without["results"]]  # the failure being guarded
    assert finding in [s.text for s, _ in with_abstract["results"]]


# ---- front matter, verbatim check, and the "never empty" guarantee ----------------


def test_strip_front_matter_removes_author_block_and_keywords():
    from app.services.paper_brief import strip_front_matter

    raw = (
        "Carlo Merola [0009 - 0000] and Jane Doe Department of Computer Science, University of "
        "Bologna carlo@unibo.it Abstract. Retrieval-augmented generation is useful. It scales "
        "well. Keywords: Retrieval - Chunking"
    )
    cleaned = strip_front_matter(raw)
    assert cleaned.startswith("Retrieval-augmented generation is useful.")
    assert "Bologna" not in cleaned and "Keywords" not in cleaned


def test_strip_front_matter_ignores_lowercase_abstract_in_body_text():
    from app.services.paper_brief import strip_front_matter

    body = "Models learn abstract concepts from data. This is ordinary body text."
    assert strip_front_matter(body) == body


def test_is_verbatim_ignores_pdf_hyphenation_spacing_and_case():
    from app.services.paper_brief import is_verbatim

    passage = "RAG has become a trans- formative approach for enhancing large language models."
    assert is_verbatim("RAG has become a transformative approach for enhancing", passage)
    assert not is_verbatim("RAG has become an outdated approach for replacing everything", passage)
    assert not is_verbatim("too short", "too short")  # trivially short strings never count


class ScriptedLLM(LLMProvider):
    """Replies to successive `complete` calls from a script; records the prompts."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate_claims(self, topic, passages, max_claims):
        return []

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply)


def _evidence_for_all_aspects():
    from app.services.paper_brief import ASPECTS, BriefItem, Evidence

    def sentence(text):
        return Sentence(
            text=text, chunk_index=0, chunk_id="c0", page=1, section=None, chunk_text=text
        )

    offered, broad, extractive = {}, {}, {}
    for aspect in ASPECTS:
        s = sentence(f"The paper states its {aspect.key} in this reasonably long sentence here.")
        offered[aspect.key] = broad[aspect.key] = [(s, 0.5)]
        extractive[aspect.key] = [
            BriefItem(aspect=aspect, text=s.text, quote=s.text, sentence=s, similarity=0.5)
        ]
    return Evidence(offered=offered, broad=broad, extractive=extractive)


def test_assemble_items_repairs_then_falls_back_so_no_aspect_is_empty():
    from app.services.paper_brief import assemble_items

    provider = ScriptedLLM(
        [
            {"problem": [{"statement": "P written by the model.", "evidence": 1}]},  # 1st pass
            {"method": [{"statement": "M written in the repair.", "evidence": 1}]},  # repair pass
        ]
    )
    items, mode = assemble_items(provider, "T", _evidence_for_all_aspects())

    by_key = {item.aspect.key: item for item in items}
    assert mode == "llm:scripted"
    assert set(by_key) == {"problem", "method", "results", "matters"}  # nothing empty
    assert by_key["problem"].text == "P written by the model."
    assert by_key["method"].text == "M written in the repair."
    assert (
        by_key["results"].text == by_key["results"].quote
    )  # last resort: the paper's own sentence
    assert by_key["matters"].text == by_key["matters"].quote
    # The repair pass only asked about the aspects that were missing.
    repair_shape = provider.prompts[1].split("Return JSON only")[1]
    assert '"problem"' not in repair_shape
    assert '"method"' in repair_shape and '"results"' in repair_shape


def test_assemble_items_survives_a_failing_llm():
    from app.services.paper_brief import assemble_items

    provider = ScriptedLLM([RuntimeError("503"), RuntimeError("503")])
    items, mode = assemble_items(provider, "T", _evidence_for_all_aspects())
    assert mode == "extractive"
    assert {item.aspect.key for item in items} == {"problem", "method", "results", "matters"}


def test_assemble_items_extractive_provider_fills_every_aspect():
    from app.services.paper_brief import assemble_items

    items, mode = assemble_items(ExtractiveProvider(), "T", _evidence_for_all_aspects())
    assert mode == "extractive"
    assert len({item.aspect.key for item in items}) == 4


def test_abstract_sentence_missing_from_pdf_sentences_is_rescued_only_if_really_in_the_pdf(
    monkeypatch,
):
    """The PDF sentence is unusable (starts lowercase after a bad split), so the near-copy
    match finds nothing - but the text is in the chunk, so the arXiv sentence is rescued.
    A sentence that is NOT in the chunk (arXiv wording differs) must be dropped, not cited."""
    from types import SimpleNamespace

    from app.services import paper_brief

    vocab = ["retrieval", "chunking", "coherence", "cost"]
    monkeypatch.setattr(
        paper_brief,
        "embed_texts",
        lambda texts: [[float(w in t.lower()) for w in vocab] + [0.01] for t in texts],
    )
    monkeypatch.setattr(paper_brief, "embed_query", lambda text: [1.0, 1.0, 1.0, 1.0, 0.01])

    in_pdf = "Retrieval improves coherence at a lower cost for chunking."
    not_in_pdf = "Completely different wording about retrieval and chunking cost only here."
    chunks = [
        SimpleNamespace(
            id="c0",
            page=1,
            section=None,
            text="and then retrieval improves coherence at a lower cost for chunking",
        )
    ]
    offered = paper_brief.select_evidence(chunks, abstract=f"{in_pdf} {not_in_pdf}").offered
    texts = {s.text for pairs in offered.values() for s, _ in pairs}
    assert in_pdf in texts
    assert not_in_pdf not in texts


# ---- search: recent papers ------------------------------------------------------


def _candidate(arxiv_id, published, score):
    from app.services.arxiv import ArxivPaper
    from app.services.topic_search import Candidate

    return Candidate(
        paper=ArxivPaper(arxiv_id, f"Paper {arxiv_id}", ["A. Author"], published, "abstract"),
        score=score,
    )


def test_pick_with_recent_share_guarantees_relevant_recent_papers():
    from datetime import date

    from app.services.topic_search import pick_with_recent_share

    since = date(2024, 9, 1)
    ranked = [
        _candidate("old1", "2019-01-01", 0.90),
        _candidate("old2", "2020-01-01", 0.89),
        _candidate("old3", "2021-01-01", 0.88),
        _candidate("old4", "2018-01-01", 0.87),
        _candidate("old5", "2017-01-01", 0.86),
        _candidate("new1", "2026-03-01", 0.85),  # recent and nearly as relevant: must be included
        _candidate("off", "2026-05-01", 0.60),  # recent but off-topic: must NOT be pulled in
    ]
    picked = pick_with_recent_share(ranked, limit=5, recent_since=since)
    ids = [c.paper.arxiv_id for c in picked]
    assert len(picked) == 5
    assert "new1" in ids and "off" not in ids
    assert [c.score for c in picked] == sorted((c.score for c in picked), reverse=True)
    assert next(c for c in picked if c.paper.arxiv_id == "new1").is_recent
    assert not next(c for c in picked if c.paper.arxiv_id == "old1").is_recent


def test_search_arxiv_recent_mode_sorts_by_date_and_filters_the_window(monkeypatch):
    from datetime import date

    from app.services import arxiv

    seen = {}

    def fake_get(url, *, params=None, timeout):
        seen.update(params)
        return type("R", (), {"text": ATOM})()

    monkeypatch.setattr(arxiv, "_get_with_retries", fake_get)
    monkeypatch.setattr(arxiv, "_throttle_api", lambda: None)
    papers = arxiv.search_arxiv(
        "efficient attention",
        sort_by="submittedDate",
        submitted_after=date(2024, 9, 1),
    )
    assert papers and seen["sortBy"] == "submittedDate"
    assert "submittedDate:[202409010000 TO" in seen["search_query"]
    assert "all:efficient AND all:attention" in seen["search_query"]
