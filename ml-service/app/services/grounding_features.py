"""Engineered features for grounding verification: (claim, passage) -> vector.

Deliberately not a from-scratch neural model - a small set of cheap,
interpretable signals (semantic similarity + lexical overlap + surface
form) fed into a lightweight classifier (see grounding_verifier.py). This
keeps the whole verifier trainable on a hand-labeled set of a few dozen
examples without needing GPU-scale data, while still being a real trained
model with measured precision/recall (see evals/run_eval.py).
"""

import re
from difflib import SequenceMatcher

import numpy as np

from app.services.embeddings import embed_texts

TOKEN_RE = re.compile(r"[a-z0-9]+")

FEATURE_NAMES = [
    "cosine_similarity",
    "claim_token_coverage",
    "jaccard_overlap",
    "sequence_ratio",
    "length_ratio",
]


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _cosine(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / denom) if denom > 0 else 0.0


def extract_features(claim: str, passage: str) -> list[float]:
    claim_embedding, passage_embedding = embed_texts([claim, passage])
    cosine = _cosine(claim_embedding, passage_embedding)

    claim_tokens = _tokens(claim)
    passage_tokens = _tokens(passage)
    overlap = claim_tokens & passage_tokens
    union = claim_tokens | passage_tokens

    token_coverage = len(overlap) / len(claim_tokens) if claim_tokens else 0.0
    jaccard = len(overlap) / len(union) if union else 0.0
    sequence_ratio = SequenceMatcher(None, claim.lower(), passage.lower()).ratio()
    length_ratio = (
        min(len(claim), len(passage)) / max(len(claim), len(passage))
        if claim and passage
        else 0.0
    )

    return [cosine, token_coverage, jaccard, sequence_ratio, length_ratio]
