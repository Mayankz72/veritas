---
description: Generate evidence-grounded claims and quiz questions for an ingested paper, and present them with their grounding labels and source pages/quotes.
disable-model-invocation: false
allowed-tools: "Bash(curl)"
---

# Analyze Paper

Runs the retrieval + claim/quiz generation + grounding-verification pipeline
against an already-ingested document (see `ingest-paper` if you don't have a
document id yet) and presents the results in a readable report.

## Steps

1. You need a **document id** (from `ingest-paper`) and a **topic/query**
   from `$ARGUMENTS`. If the user didn't give a topic, ask what they want
   covered, or suggest one of the built-in templates:

   ```bash
   curl -s "${user_config.backend_url}/templates"
   ```

2. Generate claims for that topic:

   ```bash
   curl -s -X POST "${user_config.backend_url}/documents/<DOCUMENT_ID>/claims/generate" \
     -H "Content-Type: application/json" \
     -d '{"query": "<TOPIC>", "k": 5, "max_claims": 5}'
   ```

3. Generate quiz questions for the same topic:

   ```bash
   curl -s -X POST "${user_config.backend_url}/documents/<DOCUMENT_ID>/quiz/generate" \
     -H "Content-Type: application/json" \
     -d '{"query": "<TOPIC>", "k": 5, "max_questions": 5}'
   ```

4. Present the results as a short report, not raw JSON:
   - For each claim: the claim text, its `groundingLabel` (supported /
     partial / unsupported) with `groundingScore`, the exact `quote` it's
     grounded in, and the `page` number.
   - For each quiz question: the question (with its blank), and the answer
     with page number.
   - If any claim came back `unsupported`, call that out explicitly rather
     than burying it in the list - that's the whole point of grounding
     verification.

5. The default generation provider is extractive (no LLM, no API key) unless
   the person running the ml-service set `LLM_PROVIDER` to something else -
   don't imply these are polished LLM prose; they're evidence-first, meant to
   be checked against the quote, not read as a summary.
