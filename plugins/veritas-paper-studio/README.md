# Veritas Paper Studio (Claude Code plugin)

Drives the Veritas ingestion → grounded-claim-generation → health-check
pipeline from Claude Code, using the agent's own reasoning to read results
and decide what to do next - no separate API key needed, since the
ml-service's default generation provider (`ExtractiveProvider`) doesn't call
an LLM at all, and the skills themselves just shell out to the ml-service's
REST API with `curl`.

## Prerequisites

The Veritas ml-service must be running (from the repo root):

```bash
docker compose up -d
cd ml-service && uvicorn app.main:app --reload --port 8000
```

## Install

```bash
claude plugin marketplace add <owner>/<repo>
claude plugin install veritas-paper-studio@veritas-research-tools
```

**Local development** (no install needed):

```bash
claude --plugin-dir ./plugins/veritas-paper-studio
```

## Skills

- `/veritas-paper-studio:ingest-paper <arxiv-id-or-pdf-path>` - parse and embed a paper
- `/veritas-paper-studio:analyze-paper <topic>` - generate grounded claims + quiz for a topic
- `/veritas-paper-studio:health-check <document-id>` - user-only: which sections have thin/weak evidence

`ingest-paper` and `analyze-paper` also auto-trigger when Claude judges them
relevant to what you asked (`disable-model-invocation: false`); `health-check`
is user-invoked only.

## Configuration

`backend_url` (default `http://localhost:8000`) - set this if your
ml-service runs somewhere other than localhost:8000.
