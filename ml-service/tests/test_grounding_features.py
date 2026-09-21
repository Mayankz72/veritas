"""These tests call the real embedding model (fastembed downloads
BAAI/bge-small-en-v1.5 on first use, then caches it locally/in CI cache) -
that's deliberate: the feature extractor's whole point is to reflect real
semantic similarity, so a mocked embedding would test nothing meaningful.
"""

from app.services.grounding_features import FEATURE_NAMES, extract_features


def test_feature_vector_shape():
    features = extract_features("The sky is blue.", "The sky is blue today.")
    assert len(features) == len(FEATURE_NAMES) == 5
    assert all(isinstance(f, float) for f in features)


def test_identical_text_scores_near_maximum():
    text = "Self-attention connects all positions with a constant number of operations."
    cosine, coverage, jaccard, seq_ratio, _ = extract_features(text, text)
    assert cosine > 0.99
    assert coverage == 1.0
    assert jaccard == 1.0
    assert seq_ratio == 1.0


def test_unrelated_text_scores_low():
    claim = "The stock market fell sharply on Tuesday amid inflation fears."
    passage = "Self-attention connects all positions with a constant number of operations."
    cosine, coverage, jaccard, _, _ = extract_features(claim, passage)
    assert cosine < 0.6
    assert coverage < 0.3
    assert jaccard < 0.3


def test_supported_claim_scores_higher_than_unsupported():
    passage = (
        "We used the Adam optimizer with beta_1 = 0.9, beta_2 = 0.98 and epsilon = 10^-9."
    )
    supported_claim = "The authors used the Adam optimizer with beta_1 = 0.9."
    unsupported_claim = "The authors used plain stochastic gradient descent with no momentum."

    supported_features = extract_features(supported_claim, passage)
    unsupported_features = extract_features(unsupported_claim, passage)

    # cosine similarity (index 0) should clearly separate the two
    assert supported_features[0] > unsupported_features[0]
