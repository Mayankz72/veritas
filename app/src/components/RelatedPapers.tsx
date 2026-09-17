"use client";

import { useEffect, useState } from "react";
import { getRelatedDocuments } from "@/lib/ml-service";
import type { RelatedDocument } from "@/lib/schemas/document";

/**
 * Phase 7: nearest-neighbor search over whole-document embeddings across
 * whatever else has been ingested. With only a handful of documents this
 * demonstrates the retrieval mechanism rather than a benchmarked
 * recall@k system - see ROADMAP.md for the honest before/after numbers.
 */
export function RelatedPapers({ documentId }: { documentId: string }) {
  const [related, setRelated] = useState<RelatedDocument[] | null>(null);

  useEffect(() => {
    getRelatedDocuments(documentId).then(setRelated).catch(() => setRelated([]));
  }, [documentId]);

  if (related === null || related.length === 0) return null;

  return (
    <section className="space-y-2">
      <h2 className="text-lg font-semibold">Related papers in this corpus</h2>
      <ul className="space-y-1">
        {related.map((r) => (
          <li key={r.documentId} className="flex justify-between text-sm">
            <span>{r.title}</span>
            <span className="font-mono text-black/50 dark:text-white/50">
              {r.score.toFixed(3)}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
