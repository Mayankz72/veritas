"use client";

import { useEffect, useState } from "react";
import { exportUrl, getPublication } from "@/lib/ml-service";
import type { PublicationBundle } from "@/lib/schemas/document";
import { AttentionPlayground } from "./AttentionPlayground";
import { GroundingBadge } from "./GroundingBadge";

export function PublicationView({ publicationId }: { publicationId: string }) {
  const [bundle, setBundle] = useState<PublicationBundle | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPublication(publicationId)
      .then(setBundle)
      .catch((e) => setError(String(e)));
  }, [publicationId]);

  if (error) {
    return <p className="text-red-600">{error}</p>;
  }
  if (!bundle) {
    return <p className="text-black/50 dark:text-white/50">Loading...</p>;
  }

  const { document, claims, quiz, publication } = bundle;

  return (
    <div className="space-y-8">
      <header className="space-y-1">
        <p className="text-xs uppercase tracking-wide text-black/40 dark:text-white/40">
          Published, read-only
        </p>
        <h1 className="text-2xl font-semibold">{document.title}</h1>
        <p className="text-sm text-black/50 dark:text-white/50">
          {document.pageCount} pages
          {publication.expiresAt && (
            <> &middot; expires {new Date(publication.expiresAt).toLocaleString()}</>
          )}
        </p>
        <a
          href={exportUrl(publicationId)}
          className="inline-block text-sm underline text-black/60 dark:text-white/60"
        >
          Download .veritas.json
        </a>
      </header>

      <AttentionPlayground />

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Evidence-grounded claims ({claims.length})</h2>
        <ul className="space-y-3">
          {claims.map((claim) => (
            <li
              key={claim.id}
              id={`claim-${claim.id}`}
              className="rounded-lg border border-black/10 dark:border-white/15 p-4 space-y-2 scroll-mt-4"
            >
              <div className="flex items-start justify-between gap-3">
                <p>{claim.text}</p>
                <GroundingBadge label={claim.groundingLabel} score={claim.groundingScore} />
              </div>
              <blockquote className="text-sm text-black/60 dark:text-white/60 border-l-2 border-black/15 dark:border-white/20 pl-3">
                &ldquo;{claim.quote}&rdquo; &mdash; page {claim.page}
              </blockquote>
            </li>
          ))}
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Quiz ({quiz.length})</h2>
        <ul className="space-y-3">
          {quiz.map((q) => (
            <li
              key={q.id}
              id={`quiz-${q.id}`}
              className="rounded-lg border border-black/10 dark:border-white/15 p-4 space-y-2 scroll-mt-4"
            >
              <div className="flex items-start justify-between gap-3">
                <p>{q.question}</p>
                <GroundingBadge label={q.groundingLabel} score={q.groundingScore} />
              </div>
              <p className="text-sm text-black/60 dark:text-white/60">
                Answer: {q.answer} (page {q.page})
              </p>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
