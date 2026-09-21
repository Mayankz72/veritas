"use client";

import { useEffect, useState } from "react";
import { getDocumentHealth, regenerateClaims } from "@/lib/ml-service";
import type { GroundedClaim, SectionHealth } from "@/lib/schemas/document";

/**
 * Surfaces sections with thin or weak evidence. "Strengthen" re-runs
 * generation for that section with a wider retrieval net (more chunks,
 * more claims) rather than just re-asking the same way.
 */
export function HealthPanel({
  documentId,
  onStrengthen,
}: {
  documentId: string;
  onStrengthen: (claims: GroundedClaim[]) => void;
}) {
  const [sections, setSections] = useState<SectionHealth[]>([]);
  const [strengthening, setStrengthening] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  function refresh() {
    getDocumentHealth(documentId).then(setSections).catch((e) => setError(String(e)));
  }

  useEffect(refresh, [documentId]);

  const flagged = sections.filter((s) => s.flagged);

  async function handleStrengthen(section: SectionHealth) {
    const key = section.section ?? "__none__";
    setStrengthening(key);
    setError(null);
    try {
      const query = section.section
        ? `Explain the key points of "${section.section}".`
        : "Summarize the key points of this document.";
      const claims = await regenerateClaims(documentId, {
        query,
        section: section.section,
        k: 8,
        maxClaims: 5,
      });
      onStrengthen(claims);
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setStrengthening(null);
    }
  }

  if (sections.length === 0) return null;

  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold">
        Health panel &middot; {flagged.length} of {sections.length} sections flagged
      </h2>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      {flagged.length === 0 ? (
        <p className="text-sm text-black/50 dark:text-white/50">
          No thin-evidence sections detected.
        </p>
      ) : (
        <ul className="space-y-2">
          {flagged.map((s) => {
            const key = s.section ?? "__none__";
            return (
              <li
                key={key}
                className="flex items-center justify-between gap-3 rounded border border-amber-300/60 bg-amber-50 dark:bg-amber-900/20 dark:border-amber-700/50 px-3 py-2 text-sm"
              >
                <div>
                  <span className="font-medium">{s.section ?? "(front matter)"}</span>
                  <span className="text-black/50 dark:text-white/50"> &mdash; {s.flagReason}</span>
                </div>
                <button
                  onClick={() => handleStrengthen(s)}
                  disabled={strengthening !== null}
                  className="rounded bg-amber-600 text-white px-3 py-1 text-xs font-medium disabled:opacity-50 shrink-0"
                >
                  {strengthening === key ? "Strengthening..." : "Strengthen"}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
