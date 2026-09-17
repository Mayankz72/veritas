"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { ingestArxivPaper } from "@/lib/ml-service";

const DEMO_ARXIV_ID = "1706.03762"; // "Attention Is All You Need" - the built-in example

export function IngestForm() {
  const router = useRouter();
  const [arxivId, setArxivId] = useState(DEMO_ARXIV_ID);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const doc = await ingestArxivPaper(arxivId.trim());
      router.push(`/documents/${doc.id}`);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3 w-full max-w-md">
      <label className="text-sm">
        <span className="block mb-1 text-black/60 dark:text-white/60">arXiv ID</span>
        <input
          className="w-full rounded border border-black/15 dark:border-white/20 bg-transparent px-3 py-2"
          value={arxivId}
          onChange={(e) => setArxivId(e.target.value)}
          placeholder="1706.03762"
        />
      </label>
      <button
        type="submit"
        disabled={busy}
        className="rounded-full bg-foreground text-background px-5 py-3 text-sm font-medium disabled:opacity-50"
      >
        {busy ? "Ingesting (parsing + embedding)..." : "Ingest paper"}
      </button>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      <p className="text-xs text-black/40 dark:text-white/40">
        Ingestion fetches the PDF, extracts page-accurate sections (Phase 1),
        and embeds every chunk into pgvector (Phase 2) - first run downloads
        the embedding model, so it can take ~15s.
      </p>
    </form>
  );
}
