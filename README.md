# Veritas — Evidence-Grounded Research Paper Studio

Enter a research topic. Veritas searches arXiv, reads the papers it finds,
and produces each paper's **problem, method, key results and why it
matters**, plus **how the papers fit together** — with every statement
linked to the page and quote it rests on, and checked against the paper's
own text before it's shown.

**Live**: [app-woad-nu-48.vercel.app](https://app-woad-nu-48.vercel.app)

## Topic mode: from a topic to a literature brief

```
 topic ──▶ arXiv search ──▶ rerank by embedding similarity to the topic
                                    │
                    for each chosen paper (background job, live progress)
                                    ▼
   download PDF ─▶ page-aware chunks ─▶ pgvector embeddings
                                    ▼
   score every sentence for Problem / Method / Results / Why-it-matters
                                    ▼
   write the brief, citing a source sentence for every statement
                                    ▼
   verify every statement against its source ─▶ page + quote + badge
                                    ▼
   across papers: citation graph built from the papers' own reference
   lists, timeline, shared terms, side-by-side table, "how they fit"
```

- **Search**: arXiv relevance ranking merged with a newest-first query
  (last 2 years), reranked by embedding similarity to the topic so recent
  papers aren't buried under older, more-cited ones.
- **Per paper** (background job, live progress): download PDF → page-aware
  chunks → pgvector embeddings → every sentence scored against four
  aspects (problem, method, key results, why it matters).
- **Brief**: an LLM writes each aspect from the highest-scoring evidence
  and must cite a source sentence; without an API key it falls back to the
  paper's own best sentences. Empty aspects are retried with broader
  evidence, then a verbatim quote as last resort — no blank sections.
- **Verification**: every statement is checked against its source before
  display, with page number and quote attached.
- **Cross-paper synthesis**: a citation graph (only when a title literally
  appears in another paper's text), timeline, shared terms, comparison
  table, and a written summary of how the papers relate.
- **Limits**: arXiv search is keyword-based, so unrelated-but-matching
  papers can appear (synthesis says so rather than inventing a link);
  citation detection needs a literal title match; scanned/odd PDFs weaken
  extraction.

## How claims are generated and verified

Every claim goes through four separate, testable stages rather than just
asking an LLM and trusting the answer:

- **Extraction**: PyMuPDF plus a font-size/bold-weight heuristic recovers
  page numbers and section headings, producing page-tagged chunks
  (`pdf_extraction.py`).
- **Retrieval**: each chunk is embedded (`BAAI/bge-small-en-v1.5` via
  `fastembed`/ONNX, no torch) and stored in pgvector — a real
  cosine-similarity nearest-neighbor search, not a keyword match
  (`retrieval.py`).
- **Generation**: restricted to the retrieved passages only. The default
  `ExtractiveProvider` needs no API key; OpenAI/Anthropic/Gemini/Ollama
  are opt-in via `LLM_PROVIDER` (`claim_generation.py`).
- **Verification**: a trained classifier checks each claim against its
  source passage and labels it supported/partial/unsupported before it
  reaches the UI (`grounding_verifier.py`). For LLM-written statements
  (topic mode), an entailment judge makes the final call — the classifier
  plus a "every number must appear in the source" check can only lower
  that verdict, never raise it.

## Eval numbers

| Component | Metric | Result |
|---|---|---|
| Grounding verifier | Accuracy (5-fold CV, 48-example hand-labeled set) | **70.8%** |
| Grounding verifier | Macro F1 (3-way: supported/partial/unsupported) | **0.709** |
| Retrieval | Cosine similarity, on-topic query vs. retrieved passages | **0.72–0.81** |

Full methodology and the eval harness:
[`ml-service/evals/results.md`](./ml-service/evals/results.md), reproducible
with `python -m evals.run_eval`.

## Design decisions

- **Logistic regression over 5 features for grounding, not an NLI
  cross-encoder** — explainable, trains in milliseconds, no GPU. Exact-substring
  claims short-circuit to `supported`, closing the one gap the eval set missed.
- **Claim-level versioning, not word-level diffs** — regenerating a section
  marks old claims `isCurrent=False` instead of deleting them, so every
  version stays queryable.
- **Related-papers ranking reported honestly** — tested two embedding
  pooling strategies on a 5-paper corpus; neither cleanly separated
  related from unrelated papers, so that's what's reported.
- **Production-hardened**: fixed a SIGKILL/OOM crash from `fastembed`'s
  default batch size by bounding it (`EMBEDDING_BATCH_SIZE`), and a
  Postgres NUL-byte rejection from certain PDF encodings — both with
  regression tests.

## Structure

```
app/              Next.js 16 (TypeScript, App Router) — the web app
ml-service/       FastAPI (Python) — PDF parsing, embeddings, retrieval, grounding eval
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

App runs at `http://localhost:3000` by default, ML service at
`http://localhost:8000`.

## Deployment

**Live**:

- **Web app**: [app-woad-nu-48.vercel.app](https://app-woad-nu-48.vercel.app) — Vercel
- **ML service**: [veritas-ml-service.onrender.com](https://veritas-ml-service.onrender.com) — Render
- **Database**: Neon Postgres, `vector` extension enabled

Everything is fast and reliable except embedding a brand-new paper, which
takes 4-4.5 minutes and can occasionally fail under load;
`EMBEDDING_BATCH_SIZE` bounds memory and a keep-warm ping avoids
cold-starts.

