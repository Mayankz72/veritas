# Veritas — Evidence-Grounded Research Paper Studio

Working title: **Veritas**. An interactive studio that turns research papers into
verifiable, explorable workspaces — every generated claim links back to the exact
page and quote it rests on. Inspired by `trace-research-paper-studio`, built with
our own ML core so it's a real ML/RAG project, not just an LLM wrapper.

## Stack

- **Frontend/App**: Next.js 16 (App Router), React 19, TypeScript, Zod for schema validation
- **ML Service**: Python + FastAPI — PDF parsing, embeddings, retrieval, grounding
  verification, eval harness. Talks to the Next.js app over a small internal HTTP API.
- **Storage**: Postgres (metadata, projects, versions) + pgvector (embeddings).
  SQLite acceptable for local/dev.
- **PDF processing**: Poppler / PyMuPDF for text+layout extraction.
- **Models**: pluggable LLM provider (OpenAI/Gemini/Claude/Ollama) for generation;
  a real embedding model (e.g. `bge-small` / `text-embedding-3-small`) for retrieval;
  a small trained/fine-tuned classifier for grounding verification (Phase 3).
- **Testing**: Vitest + Playwright (frontend), Pytest (ML service).

## Guiding principle

Every feature that *could* be "just prompt the LLM and trust it" should instead be:
generate → retrieve real evidence → verify claim against evidence with a measured
score → only then surface it in the UI. That verification loop is what makes this
an ML project.

---

## Phase 0 — Foundations (setup)
**Goal:** Repo scaffolded, both services running, CI in place.

- [x] Monorepo layout: `/app` (Next.js), `/ml-service` (FastAPI), `/packages/shared` (shared types/schemas)
- [x] Next.js app scaffold (TS, App Router, Tailwind or CSS modules)
- [x] FastAPI service scaffold with `/health` endpoint
- [x] Postgres + pgvector via docker-compose for local dev
- [ ] Zod schemas mirrored as Pydantic models (Pydantic side done in `ml-service/app/schemas/document.py`; Zod side lands when the frontend consumes it in a later phase)
- [x] Git repo, `.gitignore`, README, lint/format (ESLint/Prettier, ruff/black)
- [x] GitHub Actions: lint + unit tests on push

**Deliverable:** `docker-compose up` runs Postgres; `npm run dev` and `uvicorn` both boot; a trivial end-to-end ping from Next.js to FastAPI works.

---

## Phase 1 — PDF Ingestion & Extraction
**Goal:** Upload a PDF (or arXiv ID), get structured, page-accurate text + layout.

- [x] Upload flow: local file or arXiv ID → fetch → extract (raw PDF bytes are fetched/read but not yet persisted to blob storage — only extracted chunks are stored; fine for now, revisit if re-parsing with a better pipeline later matters)
- [x] Text extraction with page numbers and bounding boxes (PyMuPDF)
- [x] Section/heading segmentation (heuristic: font-size relative to median body size + bold detection; validated against the real "Attention Is All You Need" PDF — correctly recovered all section/subsection headings)
- [ ] Figure/table/equation region detection (deferred to Phase 7 — not started)
- [x] Chunking strategy for downstream retrieval (merges body blocks up to ~800 chars, flushes on heading/page/size boundaries, keeps page + section + bbox per chunk)
- [x] Store parsed doc (Postgres `documents`/`chunks` tables via SQLAlchemy, not flat JSON — gives us the FK structure Phase 2 needs for embeddings; `version` column exists but versioning logic itself is still a stub for Phase 5)

**Deliverable:** ✅ `POST /documents/arxiv` and `POST /documents/upload` return a `ParsedDocument`; `python -m scripts.ingest_demo` ingests the built-in example ("Attention Is All You Need") standalone. 16 passing tests (`pytest`) cover heading detection, chunk/page boundaries, and arXiv ID parsing using a synthetic PDF fixture (no network needed in CI).

---

## Phase 2 — Evidence-Grounded RAG Core (the centerpiece)
**Goal:** Real retrieval, not LLM-asserted citations.

- [x] Embed all chunks, store in pgvector (`BAAI/bge-small-en-v1.5` via `fastembed`/ONNX runtime — no torch dependency, 384-dim, computed automatically on ingest)
- [x] Retrieval endpoint: query/claim → top-k relevant chunks with similarity scores (`app/services/retrieval.py`, pgvector cosine distance). Verified live: querying "How does multi-head attention work?" against the real paper returns passages actually about multi-head attention at 0.72-0.81 cosine similarity.
- [x] Claim generation: pluggable `LLMProvider` interface (`app/services/claim_generation.py`) restricted to retrieved passages only. Default `ExtractiveProvider` needs no API key (deterministic, offline — used by tests/CI); `OpenAIProvider`/`AnthropicProvider`/`GeminiProvider`/`OllamaProvider` are opt-in via `LLM_PROVIDER` env var for a real LLM upgrade later.
- [x] Citation attachment: each `Claim` row stores `source_chunk_ids`, `page`, and `retrieval_score` (`POST /documents/{id}/claims/generate`)
- [x] "Exact quote" extraction: exact-substring fast path + difflib best-sentence-match fallback (`extract_supporting_quote`)

**Deliverable:** ✅ `POST /documents/{id}/claims/generate` produces claims with page, quote, and retrieval score, persisted and retrievable via `GET /documents/{id}/claims`. 21 passing tests total. `groundingLabel`/`groundingScore` are wired into the schema but stay `null` until Phase 3's verifier runs.

---

## Phase 3 — Grounding Verification & Eval Harness (the ML differentiator)
**Goal:** Prove the citations are actually correct, with a measured metric — this is what separates the project from "just an LLM wrapper."

- [x] Build a labeled eval set: 48 hand-labeled (claim, passage, label) triples — `ml-service/evals/grounding_eval_set.json` — built from real chunks of "Attention Is All You Need" (16 each of supported/unsupported/partial)
- [x] Implement a grounding verifier — engineering call: instead of a heavyweight NLI cross-encoder (torch/transformers, GPU-friendly but heavy to install and iterate on), built a `LogisticRegression` over 5 cheap engineered features (embedding cosine similarity, token coverage, Jaccard overlap, sequence-match ratio, length ratio — `app/services/grounding_features.py`). Fully explainable, trains in milliseconds, no GPU. If numbers had come in too weak this would have been the first thing to swap for a real NLI model.
- [x] Wire the verifier into the pipeline: `POST /documents/{id}/claims/generate` now sets `groundingLabel`/`groundingScore` on every claim (`app/services/grounding_verifier.py`)
- [x] Eval script reporting precision/recall/F1: `evals/run_eval.py`, 5-fold stratified cross-validation (out-of-fold predictions, honest for a 48-example set)
- [x] Tracked in `evals/results.md` — **resume bullet: grounding verifier achieves 70.8% accuracy / 0.709 macro-F1 (5-fold CV) distinguishing supported/unsupported/partial claims on a hand-labeled eval set.** Also documents a real limitation found by dogfooding against the live pipeline (verbatim-substring claims sat near the decision boundary) and the fix applied (exact-substring claims short-circuit to `supported` before ever reaching the classifier).

**Deliverable:** ✅ `python -m evals.run_eval` (from `ml-service/`) trains the verifier, prints the report, and writes `evals/results.md`. 27 passing tests total. Verified live: claims generated against the real ingested paper now carry real grounding labels, and the verbatim-substring gap found during manual testing was fixed, not just noted.

---

## Phase 4 — Interactive Learning Layer (feature parity)
**Goal:** Match Trace's interactive study features on top of our grounded evidence.

- [ ] LaTeX-to-MathML rendering for equations
- [ ] Restricted AST evaluator for formula playgrounds (no `eval`, safety-limited — reuse the design idea, implement independently)
- [ ] Parameter sliders driving playground recomputation
- [ ] "Primer" concept cards (LLM-generated, grounded via Phase 2/3 pipeline)
- [ ] Evidence-linked quiz question generation
- [ ] Quiz groundedness scorer (reuse the Phase 3 verifier: does the question's expected answer actually appear in the cited passage?) — measurable, another eval number

**Deliverable:** A working paper page with playgrounds, primers, and quizzes, all traceable to source.

---

## Phase 5 — Authoring Tools (feature parity)
**Goal:** Let a user edit/regenerate content without losing evidence guarantees.

- [ ] Section-level regeneration that re-runs Phase 2+3 pipeline, not just a raw re-prompt
- [ ] Narrative templates ("Method Walkthrough," "Results Briefing") as prompt+structure presets
- [ ] Word-level version history (diff-based, stored per section)
- [ ] "Health panel": surface sections with low grounding scores or thin evidence, offer a "Strengthen" action that re-retrieves with a wider net

**Deliverable:** Editable workspace where every edit re-validates grounding.

---

## Phase 6 — Publishing & Sharing (feature parity)
**Goal:** Shareable, exportable output.

- [ ] Unguessable share links `/p/<id>`
- [ ] Publication controls (include/exclude figures, expiration)
- [ ] Standalone JSON export format (`*.veritas.json`)
- [ ] Per-claim and per-section permalinks

**Deliverable:** A published paper page shareable via link, viewable without auth.

---

## Phase 7 — ML Enhancements (extra differentiators)
**Goal:** The features that make this unambiguously an ML project on a resume.

- [ ] Layout model upgrade: replace heuristic figure/section detection with a real model (LayoutLMv3 or a small trained classifier on section-heading detection) — report accuracy vs. the heuristic baseline
- [ ] Paper similarity / "related work" feature: embed papers (or their abstracts) and do nearest-neighbor search across an ingested corpus; report recall@k against a small hand-labeled "related papers" set
- [ ] Optional: fine-tune a small model (e.g. LoRA on an open embedding or classification model) for the grounding verifier if the off-the-shelf NLI model underperforms on your eval set

**Deliverable:** At least one component with a trained/fine-tuned model and a before/after metric comparison — the strongest resume line in the project.

---

## Phase 8 — Agent Integration (feature parity)
**Goal:** Match Trace's agent-native workflow.

- [ ] Claude Code plugin: skill that drives the ingestion → generation → verification pipeline via CLI/bridge commands
- [ ] Bridge commands for automated batch processing of papers
- [ ] Reuse existing app model credentials (no separate API key requirement)

**Deliverable:** `claude plugin install` works against this project's own marketplace manifest.

---

## Phase 9 — Polish, Testing, Deployment
**Goal:** Ship it somewhere real and make it demo-ready.

- [ ] E2E tests (Playwright) for core flows: upload → generate → verify → publish
- [ ] Unit tests for retrieval, verifier, AST evaluator (safety-critical — test the eval sandbox thoroughly)
- [ ] Deploy: Next.js on Vercel, ML service on Fly.io/Railway/Render, Postgres managed (Neon/Supabase)
- [ ] Record a demo GIF/video for the README
- [ ] Write the README with: architecture diagram, eval metrics table, and a "why this isn't just a wrapper" section

**Deliverable:** Live demo URL + polished README = the actual resume artifact.

---

## Suggested build order for a resume-first approach

If time is limited, prioritize in this order rather than strictly sequential phases:

1. Phase 0-2 (core RAG pipeline) — must-have
2. Phase 3 (eval harness) — the differentiator, don't skip
3. Phase 4 (learning layer, at least playgrounds + quizzes) — makes it demoable
4. Phase 9 (deploy + README) — makes it visible
5. Phase 7 (ML enhancement, pick one) — the standout resume line
6. Phases 5, 6, 8 — nice-to-have polish if time remains

## Metrics to track (put these in the README when done)

- Grounding verifier: precision / recall / F1 on labeled eval set
- Retrieval: recall@k for "does the right chunk show up in top-k"
- (If done) Layout model: accuracy vs. heuristic baseline
- (If done) Paper similarity: recall@k on labeled related-papers set
