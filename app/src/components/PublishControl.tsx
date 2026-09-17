"use client";

import { useState } from "react";
import { publishDocument } from "@/lib/ml-service";

const EXPIRY_OPTIONS = [
  { label: "Never", hours: null },
  { label: "24 hours", hours: 24 },
  { label: "7 days", hours: 24 * 7 },
  { label: "30 days", hours: 24 * 30 },
] as const;

export function PublishControl({ documentId }: { documentId: string }) {
  const [expiresInHours, setExpiresInHours] = useState<number | null>(null);
  const [link, setLink] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePublish() {
    setBusy(true);
    setError(null);
    try {
      const publication = await publishDocument(documentId, { expiresInHours });
      setLink(`${window.location.origin}/p/${publication.id}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-lg border border-black/10 dark:border-white/15 p-4 space-y-3">
      <h3 className="font-semibold text-sm">Publish a read-only link</h3>
      <div className="flex flex-wrap items-end gap-2">
        <label className="text-sm">
          <span className="block mb-1 text-black/60 dark:text-white/60">Expires</span>
          <select
            className="rounded border border-black/15 dark:border-white/20 bg-transparent px-3 py-2"
            value={expiresInHours ?? ""}
            onChange={(e) => setExpiresInHours(e.target.value ? Number(e.target.value) : null)}
          >
            {EXPIRY_OPTIONS.map((opt) => (
              <option key={opt.label} value={opt.hours ?? ""}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={handlePublish}
          disabled={busy}
          className="rounded border border-black/20 dark:border-white/25 px-4 py-2 text-sm font-medium disabled:opacity-50"
        >
          {busy ? "Publishing..." : "Publish"}
        </button>
      </div>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      {link && (
        <p className="text-sm">
          Shareable link:{" "}
          <a href={link} className="underline text-black/70 dark:text-white/70" target="_blank">
            {link}
          </a>
        </p>
      )}
    </div>
  );
}
