"""Trains and evaluates the grounding verifier.

Usage (from ml-service/, with the venv activated):
    python -m evals.run_eval

Reports 5-fold stratified cross-validated precision/recall/F1 (out-of-fold
predictions, since the hand-labeled set is small enough that a single
train/test split would be noisy), writes evals/results.md, then trains the
final model on the full dataset and saves it to ml_artifacts/ for the API
to load.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.grounding_features import extract_features  # noqa: E402
from app.services.grounding_verifier import DEFAULT_MODEL_PATH, GroundingVerifier  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent / "grounding_eval_set.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.md"


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_dataset(examples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([extract_features(ex["claim"], ex["passage"]) for ex in examples])
    y = np.array([ex["label"] for ex in examples])
    return X, y


def write_results_md(report: str, cm: np.ndarray, labels: list[str], n: int) -> None:
    lines = [
        "# Grounding Verifier — Eval Results",
        "",
        f"Dataset: `grounding_eval_set.json` — {n} hand-labeled (claim, passage) pairs "
        'built from the real chunks of "Attention Is All You Need" (roughly 1/3 each of '
        "supported / unsupported / partial).",
        "",
        "Metrics below are **5-fold stratified cross-validation** (out-of-fold "
        "predictions), not train-on-everything accuracy — the dataset is small enough "
        "that a single split would be noisy and optimistic.",
        "",
        "```",
        report.rstrip(),
        "```",
        "",
        f"Confusion matrix (rows = true label, columns = predicted), label order {labels}:",
        "",
        "```",
        str(cm),
        "```",
        "",
        "## Known limitation (and the fix that's actually in production)",
        "",
        "This eval set pairs each claim with a short, single passage chosen to be its "
        "*only* plausible source. In the live pipeline, `ExtractiveProvider` always "
        "produces claims that are literal substrings of their source chunk - a shape "
        "this 48-example training set under-samples. Spot-checking against the real "
        "ingested paper surfaced exactly that: verbatim-substring claims sometimes got "
        "misclassified as `unsupported` near the decision boundary (~0.55-0.60 "
        "confidence). `GroundingVerifier.predict()` now short-circuits exact-substring "
        "claims to `supported` with confidence 1.0 before ever consulting the "
        "classifier - a cheap, provably-correct rule for the one input shape the model "
        "was weakest on. The cross-validation numbers above measure the classifier "
        "alone (no short-circuit) and are the honest number for *non-verbatim* claims, "
        "which is what a real LLM provider (as opposed to the extractive default) would "
        "mostly produce.",
    ]
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    examples = load_eval_set()
    X, y = build_dataset(examples)

    verifier = GroundingVerifier()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(verifier.pipeline, X, y, cv=cv)

    labels = sorted(str(label) for label in set(y))
    report = classification_report(y, y_pred, digits=3)
    cm = confusion_matrix(y, y_pred, labels=labels)

    print(report)
    print(f"Confusion matrix (rows=true, cols=pred), label order {labels}:")
    print(cm)

    # Final model trained on all available labels, for the API to use.
    verifier.fit(X.tolist(), y.tolist())
    verifier.save()
    print(f"\nSaved trained verifier to {DEFAULT_MODEL_PATH}")

    write_results_md(report, cm, labels, len(examples))
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
