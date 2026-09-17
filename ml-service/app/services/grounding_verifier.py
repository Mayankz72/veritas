"""Trained grounding verifier: does a passage actually support a claim?

A logistic regression over the engineered features in grounding_features.py,
trained on evals/grounding_eval_set.json. See evals/run_eval.py for the
cross-validated precision/recall/F1 this achieves, and evals/results.md for
the latest numbers.
"""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from app.services.grounding_features import extract_features

_ML_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = _ML_SERVICE_ROOT / "ml_artifacts" / "grounding_verifier.joblib"


class GroundingVerifier:
    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline or make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )

    def fit(self, features: list[list[float]], labels: list[str]) -> "GroundingVerifier":
        self.pipeline.fit(features, labels)
        return self

    def predict(self, claim: str, passage: str) -> tuple[str, float]:
        # An exact-substring claim is definitionally supported - no need to
        # ask a 48-example classifier to guess at something that's checkable
        # directly. This also sidesteps a real gap in the training data: our
        # eval set under-samples verbatim-substring claims relative to how
        # often ExtractiveProvider produces them in production (see
        # evals/results.md "Known limitation").
        if claim.strip() and claim.strip().lower() in passage.lower():
            return "supported", 1.0

        features = [extract_features(claim, passage)]
        label = self.pipeline.predict(features)[0]
        confidence = float(max(self.pipeline.predict_proba(features)[0]))
        return label, confidence

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "GroundingVerifier":
        pipeline = joblib.load(path)
        return cls(pipeline=pipeline)


_singleton: GroundingVerifier | None = None
_load_attempted = False


def get_verifier() -> GroundingVerifier | None:
    """Returns the trained verifier, or None if it hasn't been trained yet
    (run `python -m evals.run_eval` from ml-service/ to produce it) - callers
    should treat a missing verifier as "grounding not checked yet", not an
    error."""
    global _singleton, _load_attempted
    if _singleton is None and not _load_attempted:
        _load_attempted = True
        try:
            _singleton = GroundingVerifier.load()
        except FileNotFoundError:
            _singleton = None
    return _singleton
