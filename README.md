# Veritas — Evidence-Grounded Research Paper Studio

Turns a research paper into an interactive, verifiable workspace: every
generated claim links back to the exact page and quote it rests on, backed by
a real retrieval + grounding-verification pipeline — not an LLM asserting
citations and hoping they're right.

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue)](.github/workflows/ci.yml)

**[Demo walkthrough video](./demo/walkthrough.webm)** — ingest → generate
grounded claims → publish → view the public read-only page, recorded from a
real Playwright run against the live app.

## Why this isn't just an LLM wrapper

Every claim goes through four real, separately-testable stages before it
reaches the UI:

```
 PDF ──(1) extract──▶ page-tagged chunks ──(2) embed & retrieve──▶ top-k passages
                                                                        │
                                            (3) generate, restricted to │
                                                those passages only     ▼
                                                                   candidate claim
                                                                        │
                                              (4) verify: does the passage ▼
                                                  actually support this claim?
                                                        (trained classifier)
                                                                        │
                                                    groundingLabel + score ▼
                                                          (shown in the UI)
```

1. **Extraction** — PyMuPDF + a font-size/bold heuristic recovers page
   numbers, sections, and page-tagged chunks (`ml-service/app/services/pdf_extraction.py`).
2. **Retrieval** — every chunk is embedded (`BAAI/bge-small-en-v1.5` via
   `fastembed`/ONNX, no torch) and stored in pgvector; a query does real
   cosine-similarity nearest-neighbor search, not a keyword match
   (`app/services/retrieval.py`).
3. **Generation** — a pluggable provider produces a claim *from the retrieved
   passages only*. The default (`ExtractiveProvider`) needs no API key at
   all — real LLM providers (OpenAI/Anthropic/Gemini/Ollama) are an opt-in
   upgrade via `LLM_PROVIDER` (`app/services/claim_generation.py`).
4. **Verification** — a small trained classifier scores the claim against
   its source passage and labels it `supported` / `partial` / `unsupported`
   *before* it's shown (`app/services/grounding_verifier.py`). This is the
   step a pure LLM-wrapper skips.

## The eval numbers (not just claimed — measured)

| Component | Metric | Result |
|---|---|---|
| Grounding verifier | Accuracy (5-fold CV, 48-example hand-labeled set) | **70.8%** |
| Grounding verifier | Macro F1 (3-way: supported/partial/unsupported) | **0.709** |
| Retrieval | Cosine similarity, on-topic query vs. retrieved passages | **0.72–0.81** (live-verified) |
| Related-papers ranking | Qualitative, 5-paper corpus | **Inconclusive** — reported honestly, not hidden (see [ROADMAP.md Phase 7](./ROADMAP.md)) |

Full methodology and the eval harness: [`ml-service/evals/results.md`](./ml-service/evals/results.md),
reproducible with `python -m evals.run_eval`. [ROADMAP.md](./ROADMAP.md) has
the phase-by-phase build log, including bugs found via integration testing
(a CORS misconfiguration, a NUL-byte PDF extraction crash, a verifier
calibration gap) and how each was fixed — not just the parts that worked.

## Structure

```
app/            Next.js 16 (TypeScript, App Router) — the web app
ml-service/     FastAPI (Python) — PDF parsing, embeddings, retrieval, grounding eval
plugins/        Claude Code plugin: ingest/analyze/health-check as agent skills
packages/shared/  Schema documentation shared between the two services
docker-compose.yml  Postgres + pgvector for local dev
```

## Local development

```bash
# 1. Database
docker compose up -d

# 2. ML service
cd ml-service
python -m venv .venv && source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000

# 3. Web app
cd app
npm install
npm run dev
```

App runs at `http://localhost:3000` by default (this dev machine's Windows
port-exclusion range forced port 4173 during development — see
`ROADMAP.md` Phase 4 — pass `-p <port>` to `next dev` if 3000 doesn't bind),
ML service at `http://localhost:8000`.

## Testing

```bash
# Backend: unit tests (no DB needed) + integration tests (needs docker compose up)
cd ml-service && pytest -v

# Frontend: unit tests
cd app && npm test

# Frontend: end-to-end (needs the ml-service + Postgres running)
cd app && npm run test:e2e
```

51 backend tests (41 unit, 10 integration against real Postgres), 9 frontend
unit tests, 1 full-flow E2E test (ingest → generate → publish → public view,
run headless against a real Chromium and the live backend — the source of
the demo video above).

## Deployment

Not yet deployed (no hosting accounts wired up for this project). The path:

- **Web app**: Vercel — zero-config for a standard Next.js app; set
  `NEXT_PUBLIC_ML_SERVICE_URL` to the deployed ml-service URL.
- **ML service**: `ml-service/Dockerfile` is built and smoke-tested; deploy it
  to Fly.io, Railway, or Render. Set `DATABASE_URL` to a managed Postgres with
  the `vector` extension available (Neon and Supabase both support it).
- **Database**: run the `CREATE EXTENSION vector` + table creation that
  `init_db()` already does automatically on startup — no separate migration
  step needed for a first deploy.

## Agent integration

`plugins/veritas-paper-studio/` is a Claude Code plugin (skills:
`ingest-paper`, `analyze-paper`, `health-check`) that drives this same
pipeline via the ml-service's REST API. See its own
[README](./plugins/veritas-paper-studio/README.md) for install/usage.
