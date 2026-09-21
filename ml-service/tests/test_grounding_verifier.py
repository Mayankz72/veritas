import tempfile
from pathlib import Path

from app.services.grounding_verifier import GroundingVerifier


def test_fit_predict_and_round_trip_through_disk():
    # Linearly separable synthetic features - this test exercises the
    # sklearn pipeline plumbing (fit/predict/save/load), not real semantics
    # (that's covered by evals/run_eval.py against the hand-labeled set).
    features = [
        [0.95, 0.9, 0.9, 0.9, 0.9],
        [0.9, 0.85, 0.85, 0.85, 0.85],
        [0.1, 0.1, 0.1, 0.1, 0.1],
        [0.05, 0.05, 0.05, 0.05, 0.05],
        [0.55, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.45, 0.45, 0.45, 0.45],
    ]
    labels = ["supported", "supported", "unsupported", "unsupported", "partial", "partial"]

    verifier = GroundingVerifier().fit(features, labels)
    predicted = verifier.pipeline.predict(features)
    assert list(predicted) == labels

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.joblib"
        verifier.save(path)
        assert path.exists()

        reloaded = GroundingVerifier.load(path)
        assert list(reloaded.pipeline.predict(features)) == labels


def test_exact_substring_claim_short_circuits_to_supported():
    # Even a verifier trained to say the opposite must be overridden by the
    # exact-substring rule - it's a correctness guarantee, not a suggestion.
    features = [[0.95, 0.9, 0.9, 0.9, 0.9], [0.05, 0.05, 0.05, 0.05, 0.05]]
    verifier = GroundingVerifier().fit(features, ["unsupported", "supported"])

    passage = "Self-attention connects all positions with a constant number of operations."
    claim = "connects all positions with a constant number of operations"

    label, confidence = verifier.predict(claim, passage)
    assert label == "supported"
    assert confidence == 1.0
