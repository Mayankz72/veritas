# Grounding Verifier — Eval Results

Dataset: `grounding_eval_set.json` — 48 hand-labeled (claim, passage) pairs built from the real chunks of "Attention Is All You Need" (roughly 1/3 each of supported / unsupported / partial).

Metrics below are **5-fold stratified cross-validation** (out-of-fold predictions), not train-on-everything accuracy — the dataset is small enough that a single split would be noisy and optimistic.

```
              precision    recall  f1-score   support

     partial      0.786     0.688     0.733        16
   supported      0.647     0.688     0.667        16
 unsupported      0.706     0.750     0.727        16

    accuracy                          0.708        48
   macro avg      0.713     0.708     0.709        48
weighted avg      0.713     0.708     0.709        48
```

Confusion matrix (rows = true label, columns = predicted), label order ['partial', 'supported', 'unsupported']:

```
[[11  3  2]
 [ 2 11  3]
 [ 1  3 12]]
```

## Known limitation (and the fix that's actually in production)

This eval set pairs each claim with a short, single passage chosen to be its *only* plausible source. In the live pipeline, `ExtractiveProvider` always produces claims that are literal substrings of their source chunk - a shape this 48-example training set under-samples. Spot-checking against the real ingested paper surfaced exactly that: verbatim-substring claims sometimes got misclassified as `unsupported` near the decision boundary (~0.55-0.60 confidence). `GroundingVerifier.predict()` now short-circuits exact-substring claims to `supported` with confidence 1.0 before ever consulting the classifier - a cheap, provably-correct rule for the one input shape the model was weakest on. The cross-validation numbers above measure the classifier alone (no short-circuit) and are the honest number for *non-verbatim* claims, which is what a real LLM provider (as opposed to the extractive default) would mostly produce.