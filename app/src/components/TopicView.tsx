"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { getTopic } from "@/lib/ml-service";
import type { BriefItem, Topic, TopicPaper, TopicSynthesis } from "@/lib/schemas/topic";
import { GroundingBadge } from "./GroundingBadge";

const POLL_MS = 3000;

const ASPECTS: { key: keyof TopicPaper["brief"]; label: string; hint: string }[] = [
  { key: "problem", label: "Problem", hint: "What gap or limitation the paper starts from" },
  { key: "method", label: "Method", hint: "What the authors propose and how it works" },
  { key: "results", label: "Key results", hint: "What was measured" },
  { key: "matters", label: "Why it matters", hint: "What it enables or changes" },
];

const STEP_LABEL: Record<TopicPaper["status"], string> = {
  pending: "waiting",
  ingesting: "downloading and indexing the PDF",
  analyzing: "writing the brief and checking it against the paper",
  done: "done",
  error: "failed",
};

function shortTitle(title: string): string {
  const head = title.split(/[:—–]| - /)[0].trim();
  return head.length >= 4 ? head : title;
}

export function TopicView({ topicId }: { topicId: string }) {
  const [topic, setTopic] = useState<Topic | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        const next = await getTopic(topicId);
        if (cancelled) return;
        setTopic(next);
        setError(null);
        if (next.status !== "done" && next.status !== "error") {
          timer = setTimeout(poll, POLL_MS);
        }
      } catch (e) {
        if (cancelled) return;
        setError(String(e));
        timer = setTimeout(poll, POLL_MS * 2); // backend restarting or briefly busy - keep trying
      }
    }
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [topicId]);

  if (!topic) {
    return error ? (
      <p className="text-red-600">{error}</p>
    ) : (
      <p className="text-black/50 dark:text-white/50">Loading...</p>
    );
  }

  const running = topic.status !== "done" && topic.status !== "error";
  const titleById = new Map(topic.papers.map((p) => [p.id, shortTitle(p.title)]));

  return (
    <div className="space-y-10">
      <header className="space-y-2">
        <Link href="/" className="text-sm underline text-black/50 dark:text-white/50">
          &larr; New topic
        </Link>
        <h1 className="text-2xl font-semibold">{topic.query}</h1>
        <p className="text-sm text-black/50 dark:text-white/50">
          Literature brief · {topic.papers.length} paper{topic.papers.length === 1 ? "" : "s"}
        </p>
      </header>

      {topic.status === "error" && (
        <p data-testid="topic-error" className="rounded-lg border border-red-300 bg-red-50 dark:bg-red-950/30 p-4 text-sm text-red-700 dark:text-red-300">
          {topic.error ?? "Something went wrong."}
        </p>
      )}

      {running && <ProgressPanel topic={topic} />}

      {topic.synthesis && <Synthesis synthesis={topic.synthesis} titleById={titleById} />}

      <section className="space-y-6">
        {topic.papers.map((paper) => (
          <PaperCard key={paper.id} paper={paper} />
        ))}
      </section>
    </div>
  );
}

function ProgressPanel({ topic }: { topic: Topic }) {
  const done = topic.papers.filter((p) => p.status === "done" || p.status === "error").length;
  const active = topic.papers.find((p) => p.status === "ingesting" || p.status === "analyzing");
  let headline = "Working...";
  if (topic.status === "searching") headline = "Searching arXiv for the best matches...";
  else if (topic.status === "synthesizing") headline = "Working out how the papers fit together...";
  else if (active) {
    headline = `Paper ${topic.papers.indexOf(active) + 1} of ${topic.papers.length}: ${STEP_LABEL[active.status]}`;
  }

  return (
    <section
      data-testid="progress"
      className="rounded-lg border border-black/10 dark:border-white/15 p-4 space-y-3"
    >
      <div className="flex items-center gap-3">
        <span className="inline-block h-3 w-3 animate-pulse rounded-full bg-emerald-500" />
        <p className="font-medium">{headline}</p>
      </div>
      {topic.papers.length > 0 && (
        <>
          <div className="h-1.5 w-full rounded bg-black/10 dark:bg-white/15">
            <div
              className="h-1.5 rounded bg-emerald-500 transition-all"
              style={{ width: `${(done / topic.papers.length) * 100}%` }}
            />
          </div>
          <ul className="space-y-1 text-sm">
            {topic.papers.map((p) => (
              <li key={p.id} className="flex justify-between gap-3">
                <span className="truncate">{shortTitle(p.title)}</span>
                <span className="text-black/50 dark:text-white/50 shrink-0">{p.status}</span>
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="text-xs text-black/40 dark:text-white/40">
        You can leave this page open, or come back later from the home page: the work keeps
        running in the background.
      </p>
    </section>
  );
}

function Synthesis({
  synthesis,
  titleById,
}: {
  synthesis: TopicSynthesis;
  titleById: Map<string, string>;
}) {
  const name = (id: string) => titleById.get(id) ?? id;
  const citations = synthesis.relationships.filter((r) => r.kind === "cites");
  const others = synthesis.relationships.filter((r) => r.kind === "similar");
  const foundational = new Set(synthesis.foundational);

  return (
    <section data-testid="synthesis" className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-xl font-semibold">How these papers fit together</h2>
        <p className="leading-relaxed">{synthesis.overview}</p>
        <p className="text-xs text-black/40 dark:text-white/40">
          {synthesis.mode.startsWith("llm:")
            ? `Written by ${synthesis.mode.slice(4)} from the papers' verified briefs and the citation links below.`
            : "Assembled from the papers' briefs and the citation links below."}
        </p>
      </div>

      {synthesis.howTheyFit.length > 0 && (
        <ul className="space-y-2">
          {synthesis.howTheyFit.map((item, i) => (
            <li
              key={i}
              className="rounded-lg border border-black/10 dark:border-white/15 p-3 text-sm space-y-1"
            >
              <p className="flex flex-wrap gap-1">
                {item.paperIds.map((id) => (
                  <span
                    key={id}
                    className="rounded-full bg-black/5 dark:bg-white/10 px-2 py-0.5 text-xs font-medium"
                  >
                    {name(id)}
                  </span>
                ))}
              </p>
              <p>{item.text}</p>
            </li>
          ))}
        </ul>
      )}

      <div className="space-y-2">
        <h3 className="font-semibold">Reading order</h3>
        <ol className="flex flex-wrap items-center gap-2 text-sm">
          {synthesis.readingOrder.map((id, i) => {
            const entry = synthesis.timeline.find((t) => t.paperId === id);
            return (
              <li key={id} className="flex items-center gap-2">
                {i > 0 && <span className="text-black/30 dark:text-white/30">&rarr;</span>}
                <span className="rounded-lg border border-black/10 dark:border-white/15 px-3 py-1.5">
                  <span className="font-medium">{name(id)}</span>
                  <span className="text-black/50 dark:text-white/50"> · {entry?.year}</span>
                  {foundational.has(id) && (
                    <span className="ml-2 rounded-full bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300 px-1.5 py-0.5 text-[10px] font-medium">
                      foundational
                    </span>
                  )}
                </span>
              </li>
            );
          })}
        </ol>
      </div>

      {synthesis.relationships.length > 0 && (
        <div className="space-y-2">
          <h3 className="font-semibold">Evidence for the links</h3>
          <ul className="space-y-2 text-sm">
            {[...citations, ...others].map((r, i) => (
              <li key={i} className="rounded-lg border border-black/10 dark:border-white/15 p-3 space-y-1">
                <p>
                  <span
                    className={`mr-2 rounded-full px-2 py-0.5 text-xs font-medium ${
                      r.kind === "cites"
                        ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
                        : "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300"
                    }`}
                  >
                    {r.kind === "cites" ? "cites" : "related, no citation"}
                  </span>
                  {r.explanation}
                </p>
                {r.snippet && (
                  <blockquote className="text-xs text-black/55 dark:text-white/55 border-l-2 border-black/15 dark:border-white/20 pl-3">
                    &ldquo;{r.snippet}&rdquo; &mdash; page {r.page}
                  </blockquote>
                )}
              </li>
            ))}
          </ul>
          <p className="text-xs text-black/40 dark:text-white/40">
            Citation links are found by matching a paper&rsquo;s title inside another
            paper&rsquo;s own text (its reference list); nothing here is guessed by a model. Papers
            can still be related through work outside this set.
          </p>
        </div>
      )}

      <div className="space-y-2">
        <h3 className="font-semibold">Side by side</h3>
        <div className="overflow-x-auto rounded-lg border border-black/10 dark:border-white/15">
          <table className="w-full text-sm text-left">
            <thead className="bg-black/[0.03] dark:bg-white/[0.06]">
              <tr>
                <th className="p-3 font-medium">Paper</th>
                <th className="p-3 font-medium">Problem</th>
                <th className="p-3 font-medium">Method</th>
                <th className="p-3 font-medium">Key result</th>
              </tr>
            </thead>
            <tbody>
              {synthesis.comparison.map((row) => (
                <tr key={row.paperId} className="border-t border-black/10 dark:border-white/15 align-top">
                  <td className="p-3 min-w-36">
                    <span className="font-medium">{shortTitle(row.title)}</span>
                    <span className="block text-xs text-black/50 dark:text-white/50">{row.year}</span>
                  </td>
                  <td className="p-3 min-w-56">{row.problem}</td>
                  <td className="p-3 min-w-56">{row.method}</td>
                  <td className="p-3 min-w-56">{row.results}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function PaperCard({ paper }: { paper: TopicPaper }) {
  return (
    <article
      data-testid="paper-card"
      className="rounded-xl border border-black/10 dark:border-white/15 p-5 space-y-4"
    >
      <header className="space-y-1">
        <h2 className="text-lg font-semibold leading-snug">{paper.title}</h2>
        <p className="text-xs text-black/50 dark:text-white/50">
          {paper.authors.slice(0, 4).join(", ")}
          {paper.authors.length > 4 ? " et al." : ""} · {paper.published.slice(0, 4)} ·{" "}
          <a
            className="underline"
            href={`https://arxiv.org/abs/${paper.arxivId}`}
            target="_blank"
            rel="noreferrer"
          >
            arXiv:{paper.arxivId}
          </a>
          {paper.documentId && (
            <>
              {" "}
              ·{" "}
              <Link className="underline" href={`/documents/${paper.documentId}`}>
                open full workspace
              </Link>
            </>
          )}
        </p>
      </header>

      {paper.status === "error" && (
        <p className="text-sm text-red-600">Could not process this paper: {paper.error}</p>
      )}
      {(paper.status === "pending" || paper.status === "ingesting" || paper.status === "analyzing") && (
        <p className="text-sm text-black/50 dark:text-white/50">{STEP_LABEL[paper.status]}...</p>
      )}

      {paper.status === "done" && (
        <>
          <div className="grid gap-4 sm:grid-cols-2">
            {ASPECTS.map((aspect) => (
              <div key={aspect.key} className="space-y-2">
                <h3 className="text-sm font-semibold" title={aspect.hint}>
                  {aspect.label}
                </h3>
                {paper.brief[aspect.key].length === 0 && (
                  <p className="text-sm text-black/40 dark:text-white/40">
                    Not enough clear evidence in the paper for this.
                  </p>
                )}
                {paper.brief[aspect.key].map((item) => (
                  <BriefStatement key={item.id} item={item} />
                ))}
              </div>
            ))}
          </div>
          <p className="text-xs text-black/40 dark:text-white/40">
            {paper.briefMode?.startsWith("llm:")
              ? `Written by ${paper.briefMode.slice(4)} from sentences in the paper, then checked against the source text.`
              : "Sentences taken verbatim from the paper."}
          </p>
        </>
      )}
    </article>
  );
}

function BriefStatement({ item }: { item: BriefItem }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="space-y-1">
      <p className="text-sm leading-relaxed">{item.text}</p>
      <div className="flex flex-wrap items-center gap-2">
        <GroundingBadge label={item.groundingLabel} score={item.groundingScore} />
        {item.text === item.quote && (
          <span
            className="rounded-full bg-black/5 dark:bg-white/10 px-2 py-0.5 text-xs text-black/60 dark:text-white/60"
            title="This is the paper's own sentence, not a paraphrase"
          >
            direct quote
          </span>
        )}
        <button
          onClick={() => setOpen((o) => !o)}
          className="text-xs underline text-black/50 dark:text-white/50"
        >
          {open ? "Hide source" : `Source: page ${item.page}`}
        </button>
      </div>
      {open && (
        <blockquote className="text-xs text-black/60 dark:text-white/60 border-l-2 border-black/15 dark:border-white/20 pl-3">
          &ldquo;{item.quote}&rdquo; &mdash; page {item.page}
        </blockquote>
      )}
    </div>
  );
}
