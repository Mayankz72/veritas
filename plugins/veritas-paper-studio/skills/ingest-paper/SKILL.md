---
description: Ingest a research paper (by arXiv ID or local PDF path) into the Veritas ml-service - extracts page-accurate sections and embeds every chunk for retrieval.
disable-model-invocation: false
allowed-tools: "Bash(curl)"
---

# Ingest Paper

Ingests a paper into the running Veritas ml-service (`${user_config.backend_url}`,
default `http://localhost:8000`) so later skills (`analyze-paper`, `health-check`)
have something to work with. Needs no API key - ingestion is pure PDF
parsing + a local embedding model, no LLM call.

## Steps

1. Determine what the user gave you as `$ARGUMENTS`:
   - Looks like an arXiv ID (e.g. `1706.03762`, optionally with a `v\d+` suffix,
     or a full `arxiv.org/abs/...`/`arxiv.org/pdf/...` URL): use the arXiv route.
   - Looks like a local file path ending in `.pdf`: use the upload route.
   - If neither is clear, ask the user which they meant before proceeding.

2. **arXiv route** - extract the bare ID (e.g. `1706.03762`) and call:

   ```bash
   curl -s -X POST "${user_config.backend_url}/documents/arxiv" \
     -H "Content-Type: application/json" \
     -d "{\"arxiv_id\": \"<ID>\"}"
   ```

3. **Local PDF route**:

   ```bash
   curl -s -X POST "${user_config.backend_url}/documents/upload" \
     -F "file=@<PATH_TO_PDF>"
   ```

4. The response is a `ParsedDocument` JSON: `id`, `title`, `pageCount`, `chunks`
   (an array - don't print it in full, it's usually 40-100+ entries). Report to
   the user:
   - The document title and page count
   - The number of chunks extracted
   - **The document `id`** - tell the user to keep it handy (or that you'll
     remember it for them) since `analyze-paper` and `health-check` both need it

5. If the request fails, check the obvious things before asking the user to
   debug: is the ml-service actually running at that URL (`curl -s
   ${user_config.backend_url}/health`), and is Postgres up (`docker compose up
   -d` from the repo root)?
