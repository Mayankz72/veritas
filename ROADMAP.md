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

- [x] LaTeX rendering via KaTeX (`app/src/components/Formula.tsx`) — KaTeX emits HTML plus a MathML accessibility tree by default, satisfying this without a separate MathML pipeline
- [x] Restricted AST evaluator for formula playgrounds — hand-rolled recursive-descent parser + tree-walking evaluator (`app/src/lib/safe-math.ts`), no `eval`/`new Function`, allowlisted functions only. Caught a real bug via its own test suite: a plain-object function/variable lookup let `constructor(1)` and `__proto__` resolve through the JS prototype chain instead of being rejected — fixed with a `Map` for functions and `hasOwnProperty` checks for variables (see `test_...never executes arbitrary JS` in `safe-math.test.ts`)
- [x] Parameter sliders driving playground recomputation — `AttentionPlayground.tsx` visualizes the paper's 3.2.1 scaling factor (`dot / sqrt(d_k)`) live as you drag `d_k`/dot-product sliders
- [ ] "Primer" concept cards as a distinct pedagogical UI — deferred; the grounded-claims list (below) covers the same evidence-linked-content need for now, a separate "primer" framing didn't make the cut this pass
- [x] Evidence-linked quiz question generation — cloze-deletion generator (`app/services/quiz_generation.py`), no LLM needed by default, same extractive-first philosophy as claim generation
- [x] Quiz groundedness scorer — reuses the Phase 3 verifier directly (`POST /documents/{id}/quiz/generate` sets `groundingLabel`/`groundingScore` per question)

**Deliverable:** ✅ `/documents/[id]` page: document header, interactive attention-scaling playground, a query box that generates grounded claims and quiz questions against the live ml-service, both rendered with grounding badges. Verified: production build succeeds, `tsc --noEmit` clean, ESLint clean, 9 passing Vitest tests (safe-math), 31 passing pytest tests (backend). CORS bug found and fixed during integration testing (see below). **Not verified**: actual browser rendering/interaction — the Claude-in-Chrome extension wasn't connected in this session, so sliders/KaTeX/click-through were checked via curl simulation of the exact requests the page makes (200s, correct CORS headers) and static analysis, not a real browser. Recommend a manual click-through before treating this as fully done.

**Bug found and fixed via manual integration testing:** the ml-service's CORS config only allowed `http://localhost:3000`, but this machine's Windows `netsh` dynamic port-exclusion range covers 3000 *and* 3100, so `next dev` silently lands on a different port (4173 here) — which CORS then rejected. Fixed by switching to `allow_origin_regex` matching any localhost/127.0.0.1 port.

---

## Phase 5 — Authoring Tools (feature parity)
**Goal:** Let a user edit/regenerate content without losing evidence guarantees.

- [x] Section-level regeneration that re-runs Phase 2+3 pipeline — `POST /documents/{id}/claims/regenerate` re-retrieves, re-generates, and re-verifies grounding from scratch, it doesn't just re-prompt over stale evidence
- [x] Narrative templates ("Method Walkthrough," "Results Briefing," "Architecture Overview," "Limitations & Future Work") as query presets — `GET /templates`, surfaced as a dropdown in the query box
- [ ] Word-level version history (diff-based) — engineering call: implemented as **claim-level** version history instead (`version`/`isCurrent` on `Claim`; regenerating a section marks old claims `isCurrent=False` rather than deleting them, all versions stay queryable via `?current_only=false`). A real word-level diff view is future work if claims start being hand-edited rather than only regenerated wholesale.
- [x] "Health panel": surfaces thin-evidence sections (no claims yet) and weak ones (any unsupported claim, or average grounding confidence below 0.6) with a one-click "Strengthen" that regenerates with a wider retrieval net (`k=8` vs. the default 5) — `app/services/health.py`, `GET /documents/{id}/health`, `HealthPanel.tsx`

**Deliverable:** ✅ Verified live end-to-end: generated 2 claims for "5.3 Optimizer" (health panel correctly unflagged it, avg score 1.0), then regenerated with a different query — old claims flipped to `isCurrent=False` (still queryable), 3 new v2 claims created and marked current, health panel reflected the new state. 5 new passing tests (`test_health.py`) plus the existing 31 = 36 total. Frontend: `tsc`/ESLint clean, production build succeeds, 9 Vitest tests still pass. Also fixed a stale Pydantic `class Config` deprecation warning while in the area.

---

## Phase 6 — Publishing & Sharing (feature parity)
**Goal:** Shareable, exportable output.

- [x] Unguessable share links `/p/<id>` — `id` is a `secrets.token_urlsafe(16)` slug (~130 bits), the only "auth" a publication has (matches this project's explicit no-accounts scope)
- [x] Publication controls: `include_figures` flag stored (figure *extraction* itself is still Phase 7/future work, so this is a placeholder toggle for now) and optional expiration (`expires_in_hours` → `expires_at`), enforced server-side with a `410 Gone` on fetch/export past expiry — verified live with a deliberately-already-expired publication
- [x] Standalone JSON export: `GET /publications/{id}/export` returns a `*.veritas.json` file (`Content-Disposition: attachment`) bundling the document, current claims, and quiz under a versioned `"format": "veritas-export-v1"` envelope
- [x] Per-claim and per-section permalinks — public view renders each claim/question with a stable `id="claim-<id>"` / `id="quiz-<id>"` anchor

**Deliverable:** ✅ `POST /documents/{id}/publish` → shareable `/p/<id>` link, viewable with no auth (`PublicationView.tsx`, read-only: document header, the same attention playground, claims, quiz, and a "Download .veritas.json" link). Verified end-to-end: publish → fetch bundle (3 claims, 3 quiz questions) → export downloads with correct filename/content-type → a deliberately-expired publication correctly 410s instead of serving stale content. `tsc`/ESLint clean, production build succeeds (new `/p/[id]` route), 9 Vitest + 36 pytest tests still pass.

---

## Phase 7 — ML Enhancements (extra differentiators)
**Goal:** The features that make this unambiguously an ML project on a resume.

- [ ] Layout model upgrade — **skipped deliberately**: the Phase 1 heuristic already recovers 100% of the real headings/subheadings on every paper ingested so far (verified across 5 real arXiv PDFs, not just the one demo paper). A trained classifier would also just repeat Phase 3's "logistic regression over engineered features" pattern rather than demonstrate something new. Revisit if a paper with unusual layout (multi-column, heavy figures) actually breaks the heuristic.
- [x] Paper similarity / "related work" feature — `Document.embedding` (pgvector), `app/services/related_documents.py` (cosine nearest-neighbor), `GET /documents/{id}/related`, `RelatedPapers.tsx`. Ingested a real 5-paper corpus to test it: "Attention Is All You Need" plus two of its own cited references (Layer Normalization, Adam) and two topically-adjacent-but-uncited papers (BERT, GANs).
- [ ] Recall@k against a hand-labeled set — not attempted; 5 documents is too small a corpus for a recall@k number to mean anything statistically. Reported qualitative rankings instead (below), which is the honest thing to do at this scale rather than compute a metric that would look precise but isn't meaningful.

**What actually happened (worth reading before trusting this feature):** tried two document-embedding strategies and got two different, both defensible-looking, but disagreeing rankings:
- **Whole-document mean-pool** (average every chunk's embedding): BERT 0.962, Layer Norm 0.947, GANs 0.940, Adam 0.920 — intuitively reasonable ordering (BERT closest, a direct Transformer descendant), but a narrow 0.04 band suggests the ranking is barely distinguishing anything; likely because averaging in every reference/boilerplate chunk dilutes the signal.
- **Abstract-only** (the standard practice in paper-recommendation literature - see `_document_embedding` in `app/routers/documents.py`): Adam 0.855, BERT 0.816, GANs 0.809, Layer Norm 0.804 — Adam ranking above BERT is *not* intuitively right, and the band is still narrow (~0.05).

Shipped the abstract-only version since it's the principled default per the literature, but the honest conclusion is: **at this corpus size (5 docs) with a general-purpose small embedding model (`bge-small`, 384-dim), neither pooling strategy cleanly separates related from unrelated papers.** Likely fixes, left as future work: a larger corpus so rankings average out per-document noise, a stronger/domain-tuned embedding model, or pooling that weights title+abstract+conclusion rather than abstract alone. This is reported instead of quietly picking whichever run looked better - the eval discipline from Phase 3 applied to a feature that didn't pan out as cleanly.

**Also found and fixed via this corpus ingestion:** two of the five real papers (Layer Normalization, Adam) 500'd on ingest with `PostgreSQL text fields cannot contain NUL (0x00) bytes` - some PDFs' font/ligature encoding produces literal NUL bytes during text extraction. Fixed in `_clean_text` (`app/services/pdf_extraction.py`) with a regression test.

**Deliverable:** Partial — related-papers retrieval mechanism is real and working (verified live against 5 real ingested papers), but the "strongest resume line" bar this phase set for itself isn't met: the honest result is a negative/inconclusive one on ranking quality, not a clean before/after win. The grounding verifier (Phase 3) remains the strongest trained-model resume line in this project.

---

## Phase 8 — Agent Integration (feature parity)
**Goal:** Match Trace's agent-native workflow.

- [x] Claude Code plugin: `plugins/veritas-paper-studio/` with 3 skills (`ingest-paper`, `analyze-paper`, `health-check`) that shell out to the ml-service's own REST API via `curl` — the pipeline itself already exists (Phases 1-5), this phase just exposes it as agent-invocable commands rather than building a parallel implementation
- [ ] Bridge commands for automated *batch* processing of papers — not built; the skills operate on one paper per invocation. A batch-ingest skill (loop over a list of arXiv IDs) would be a small follow-up on top of `ingest-paper`, not new infrastructure.
- [x] Reuse existing agent credentials, no separate API key: skills call `curl` directly, and the ml-service's default generation provider (`ExtractiveProvider`) needs no LLM/API key at all — the strongest version of "no separate API key requirement," since even the backend generation step is keyless by default
- [x] Plugin marketplace manifest at the repo root (`.claude-plugin/marketplace.json`) referencing the plugin via a local relative-path source, so `claude --plugin-dir ./plugins/veritas-paper-studio` works today for local dev, and `claude plugin marketplace add <owner>/<repo>` + `claude plugin install veritas-paper-studio@veritas-research-tools` once pushed to GitHub

**How this was verified (not just written and assumed correct):** consulted a specialized guide agent for the authoritative plugin/marketplace JSON schema and SKILL.md frontmatter (rather than guessing a plausible-looking format), then ran `claude plugin validate` — the actual CLI's own schema checker — against the marketplace manifest, the plugin manifest, and the skills directory. All three passed clean (one marketplace-description warning, fixed).

**Deliverable:** ✅ `claude plugin validate .` (marketplace), `claude plugin validate ./plugins/veritas-paper-studio` (plugin manifest), and `claude plugin validate ./plugins/veritas-paper-studio/skills` (all 3 skills) all pass. Not yet verified: an actual end-to-end `/veritas-paper-studio:analyze-paper` invocation inside a live Claude Code session driving the real ml-service (would need a second session to install/invoke it) — the skill bodies were written to match the real API contracts from Phases 1-6 (verified endpoint shapes), but that specific integration path is untested.

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
