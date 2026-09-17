---
description: Report which sections of an ingested paper have thin or weak evidence coverage.
disable-model-invocation: true
allowed-tools: "Bash(curl)"
---

# Health Check

User-only diagnostic (not auto-triggered): surfaces sections of an ingested
document that have no claims yet, or whose current claims are weakly
grounded, so the user knows where to run `analyze-paper` next rather than
assuming a document is fully covered just because *some* claims exist.

## Steps

1. Call the health endpoint for the document id in `$ARGUMENTS`:

   ```bash
   curl -s "${user_config.backend_url}/documents/<DOCUMENT_ID>/health"
   ```

2. Each entry has `section`, `claimCount`, `supportedCount`/`partialCount`/
   `unsupportedCount`, `averageGroundingScore`, `flagged`, and `flagReason`.
   Report only the **flagged** sections to the user, grouped by reason (no
   claims yet vs. weak grounding), and mention the total count (e.g. "6 of 27
   sections flagged").

3. For a flagged section the user wants addressed, suggest re-running
   `analyze-paper` with that section's name/topic as the query, or call the
   regenerate endpoint directly with a wider net:

   ```bash
   curl -s -X POST "${user_config.backend_url}/documents/<DOCUMENT_ID>/claims/regenerate" \
     -H "Content-Type: application/json" \
     -d '{"query": "<SECTION_TOPIC>", "section": "<SECTION_NAME>", "k": 8, "max_claims": 5}'
   ```

   Regeneration keeps prior claims in the database (marked superseded, not
   deleted) rather than losing history - see the `version`/`isCurrent`
   fields if the user asks what happened to the old ones.
