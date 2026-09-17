# Shared Schemas

Single source of truth documentation for data shapes used by both the Next.js
app (`/app`, validated with Zod) and the ML service (`/ml-service`, validated
with Pydantic). TypeScript and Python don't share a runtime, so schemas are
kept in sync manually — this folder documents the canonical shape; each side
implements its own validator matching it.

## `ParsedDocument` (Phase 1)

```ts
type ParsedDocument = {
  id: string;
  sourceType: "upload" | "arxiv";
  title: string;
  pageCount: number;
  chunks: Chunk[];
};

type Chunk = {
  id: string;
  page: number;
  section: string | null;
  text: string;
  bbox: [number, number, number, number] | null;
};
```

## `GroundedClaim` (Phase 2/3)

```ts
type GroundedClaim = {
  id: string;
  documentId: string;
  section: string | null;
  text: string;
  sourceChunkIds: string[];
  page: number;
  quote: string;
  retrievalScore: number;
  groundingLabel: "supported" | "unsupported" | "partial" | null;
  groundingScore: number | null;
};
```

Update this file whenever a schema changes on either side, and update both
implementations (`app/src/lib/schemas/*.ts` and `ml-service/app/schemas/*.py`)
to match.
