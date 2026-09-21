"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { createTopic, deleteTopic, listTopics, searchTopic } from "@/lib/ml-service";
import type { ArxivCandidate, TopicSummary } from "@/lib/schemas/topic";

const EXAMPLES = ["efficient attention for long sequences", "retrieval-augmented generation", "diffusion models"];

export function TopicSearch() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [candidates, setCandidates] = useState<ArxivCandidate[] | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState<"search" | "build" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState<TopicSummary[]>([]);
  const [confirmingDelete, setConfirmingDelete] = useState<string | null>(null);

  useEffect(() => {
    listTopics().then(setRecent).catch(() => {});
  }, []);

  async function handleSearch(e?: React.FormEvent, override?: string) {
    e?.preventDefault();
    const topic = (override ?? query).trim();
    if (!topic) return;
    setQuery(topic);
    setBusy("search");
    setError(null);
    setCandidates(null);
    try {
      const found = await searchTopic(topic, 6);
      setCandidates(found);
      setSelected(new Set(found.slice(0, 5).map((c) => c.arxivId)));
      if (found.length === 0) setError("No papers found on arXiv for that topic. Try broader words.");
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(null);
    }
  }

  async function handleBuild() {
    if (!candidates) return;
    setBusy("build");
    setError(null);
    try {
      const papers = candidates.filter((c) => selected.has(c.arxivId));
      const topic = await createTopic({ query, papers });
      router.push(`/topics/${topic.id}`);
    } catch (err) {
      setError(String(err));
      setBusy(null);
    }
  }

  async function handleDelete(id: string) {
    if (confirmingDelete !== id) {
      setConfirmingDelete(id); // first click arms it; second click deletes
      return;
    }
    setConfirmingDelete(null);
    try {
      await deleteTopic(id);
      setRecent((prev) => prev.filter((t) => t.id !== id));
    } catch (err) {
      setError(String(err));
    }
  }

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <div className="w-full max-w-2xl space-y-5 text-left">
      <form onSubmit={handleSearch} className="flex gap-2">
        <input
          data-testid="topic-input"
          className="flex-1 rounded-full border border-black/15 dark:border-white/20 bg-transparent px-5 py-3"
          placeholder="Enter a research topic, e.g. efficient attention for long sequences"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          type="submit"
          disabled={busy !== null || !query.trim()}
          className="rounded-full bg-foreground text-background px-6 py-3 text-sm font-medium disabled:opacity-50"
        >
          {busy === "search" ? "Searching arXiv..." : "Find papers"}
        </button>
      </form>

      {candidates === null && busy === null && (
        <p className="text-sm text-black/50 dark:text-white/50">
          Try:{" "}
          {EXAMPLES.map((example, i) => (
            <span key={example}>
              {i > 0 && " · "}
              <button
                type="button"
                className="underline"
                onClick={() => handleSearch(undefined, example)}
              >
                {example}
              </button>
            </span>
          ))}
        </p>
      )}

      {error && <p className="text-red-600 text-sm">{error}</p>}

      {candidates && candidates.length > 0 && (
        <section className="space-y-3" data-testid="candidates">
          <div className="flex items-baseline justify-between">
            <h2 className="font-semibold">Papers found for &ldquo;{query}&rdquo;</h2>
            <span className="text-xs text-black/50 dark:text-white/50">
              relevance to your topic, plus recent work
            </span>
          </div>
          <ul className="space-y-2">
            {candidates.map((c) => (
              <li key={c.arxivId}>
                <label className="flex gap-3 rounded-lg border border-black/10 dark:border-white/15 p-3 cursor-pointer hover:bg-black/[0.02] dark:hover:bg-white/[0.04]">
                  <input
                    type="checkbox"
                    className="mt-1"
                    checked={selected.has(c.arxivId)}
                    onChange={() => toggle(c.arxivId)}
                  />
                  <span className="space-y-1 min-w-0">
                    <span className="block font-medium leading-snug">
                      {c.title}
                      {c.isRecent && (
                        <span className="ml-2 align-middle rounded-full bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-300 px-1.5 py-0.5 text-[10px] font-medium">
                          new
                        </span>
                      )}
                    </span>
                    <span className="block text-xs text-black/50 dark:text-white/50">
                      {c.authors.slice(0, 3).join(", ")}
                      {c.authors.length > 3 ? " et al." : ""} · {c.published.slice(0, 4)} ·
                      arXiv:{c.arxivId} · match {(c.score * 100).toFixed(0)}%
                    </span>
                    <span className="text-sm text-black/65 dark:text-white/65 line-clamp-2">
                      {c.abstract}
                    </span>
                  </span>
                </label>
              </li>
            ))}
          </ul>
          <button
            data-testid="build-brief"
            onClick={handleBuild}
            disabled={busy !== null || selected.size === 0}
            className="w-full rounded-full bg-foreground text-background px-5 py-3 text-sm font-medium disabled:opacity-50"
          >
            {busy === "build"
              ? "Starting..."
              : `Build literature brief from ${selected.size} paper${selected.size === 1 ? "" : "s"}`}
          </button>
          <p className="text-xs text-black/40 dark:text-white/40">
            Each paper is downloaded and read page by page, summarised as Problem / Method / Key
            results / Why it matters, and every statement is checked against the paper&rsquo;s own
            text. Takes a few minutes; you can watch it progress.
          </p>
        </section>
      )}

      {recent.length > 0 && candidates === null && (
        <section className="space-y-2 pt-2">
          <h2 className="text-sm font-semibold text-black/60 dark:text-white/60">Recent briefs</h2>
          <ul className="space-y-1">
            {recent.slice(0, 10).map((t) => (
              <li key={t.id} className="flex items-center justify-between gap-3 text-sm">
                <Link href={`/topics/${t.id}`} className="underline truncate">
                  {t.query}
                </Link>
                <span className="flex items-center gap-3 shrink-0 text-black/40 dark:text-white/40">
                  <span>
                    {t.paperCount} papers · {t.status}
                  </span>
                  <button
                    onClick={() => handleDelete(t.id)}
                    onBlur={() => setConfirmingDelete((c) => (c === t.id ? null : c))}
                    className={
                      confirmingDelete === t.id
                        ? "rounded bg-red-600 px-2 py-0.5 text-xs font-medium text-white"
                        : "text-xs underline hover:text-red-600"
                    }
                    aria-label={
                      confirmingDelete === t.id
                        ? `Confirm delete: ${t.query}`
                        : `Delete brief: ${t.query}`
                    }
                  >
                    {confirmingDelete === t.id ? "Confirm delete" : "Delete"}
                  </button>
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
