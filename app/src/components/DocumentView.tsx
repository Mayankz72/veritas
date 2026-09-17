"use client";

import { useEffect, useState } from "react";
import {
  generateClaims,
  generateQuiz,
  getDocument,
  listClaims,
  listQuiz,
  listTemplates,
} from "@/lib/ml-service";
import type {
  GroundedClaim,
  NarrativeTemplate,
  ParsedDocument,
  QuizQuestion,
} from "@/lib/schemas/document";
import { AttentionPlayground } from "./AttentionPlayground";
import { GroundingBadge } from "./GroundingBadge";
import { HealthPanel } from "./HealthPanel";
import { PublishControl } from "./PublishControl";
import { RelatedPapers } from "./RelatedPapers";

export function DocumentView({ documentId }: { documentId: string }) {
  const [document, setDocument] = useState<ParsedDocument | null>(null);
  const [claims, setClaims] = useState<GroundedClaim[]>([]);
  const [quiz, setQuiz] = useState<QuizQuestion[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("How does multi-head attention work?");
  const [busy, setBusy] = useState<"claims" | "quiz" | null>(null);
  const [templates, setTemplates] = useState<NarrativeTemplate[]>([]);

  useEffect(() => {
    getDocument(documentId).then(setDocument).catch((e) => setError(String(e)));
    listClaims(documentId).then(setClaims).catch(() => {});
    listQuiz(documentId).then(setQuiz).catch(() => {});
    listTemplates().then(setTemplates).catch(() => {});
  }, [documentId]);

  async function handleGenerateClaims() {
    setBusy("claims");
    setError(null);
    try {
      const newClaims = await generateClaims(documentId, { query, k: 5, maxClaims: 5 });
      setClaims((prev) => [...newClaims, ...prev]);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  }

  async function handleGenerateQuiz() {
    setBusy("quiz");
    setError(null);
    try {
      const newQuestions = await generateQuiz(documentId, { query, k: 5, maxQuestions: 5 });
      setQuiz((prev) => [...newQuestions, ...prev]);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  }

  if (!document) {
    return error ? (
      <p className="text-red-600">{error}</p>
    ) : (
      <p className="text-black/50 dark:text-white/50">Loading document...</p>
    );
  }

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-semibold">{document.title}</h1>
        <p className="text-sm text-black/50 dark:text-white/50">
          {document.pageCount} pages &middot; {document.chunks.length} chunks &middot;{" "}
          {document.sourceType}
        </p>
      </header>

      <AttentionPlayground />

      <RelatedPapers documentId={documentId} />

      <PublishControl documentId={documentId} />

      <HealthPanel
        documentId={documentId}
        onStrengthen={(newClaims) => setClaims((prev) => [...newClaims, ...prev])}
      />

      <section className="space-y-3">
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex-1 min-w-64 text-sm">
            <span className="block mb-1 text-black/60 dark:text-white/60">
              Query / section topic
            </span>
            <input
              className="w-full rounded border border-black/15 dark:border-white/20 bg-transparent px-3 py-2"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          {templates.length > 0 && (
            <label className="text-sm">
              <span className="block mb-1 text-black/60 dark:text-white/60">Template</span>
              <select
                className="rounded border border-black/15 dark:border-white/20 bg-transparent px-3 py-2"
                defaultValue=""
                onChange={(e) => {
                  const t = templates.find((tpl) => tpl.key === e.target.value);
                  if (t) setQuery(t.query);
                }}
              >
                <option value="" disabled>
                  Choose a template...
                </option>
                {templates.map((t) => (
                  <option key={t.key} value={t.key}>
                    {t.label}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button
            onClick={handleGenerateClaims}
            disabled={busy !== null}
            className="rounded bg-foreground text-background px-4 py-2 text-sm font-medium disabled:opacity-50"
          >
            {busy === "claims" ? "Generating..." : "Generate claims"}
          </button>
          <button
            onClick={handleGenerateQuiz}
            disabled={busy !== null}
            className="rounded border border-black/20 dark:border-white/25 px-4 py-2 text-sm font-medium disabled:opacity-50"
          >
            {busy === "quiz" ? "Generating..." : "Generate quiz"}
          </button>
        </div>
        {error && <p className="text-red-600 text-sm">{error}</p>}
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">
          Evidence-grounded claims ({claims.length})
        </h2>
        {claims.length === 0 && (
          <p className="text-sm text-black/50 dark:text-white/50">
            No claims yet - generate some above.
          </p>
        )}
        <ul className="space-y-3">
          {claims.map((claim) => (
            <li
              key={claim.id}
              className="rounded-lg border border-black/10 dark:border-white/15 p-4 space-y-2"
            >
              <div className="flex items-start justify-between gap-3">
                <p>{claim.text}</p>
                <GroundingBadge label={claim.groundingLabel} score={claim.groundingScore} />
              </div>
              <blockquote className="text-sm text-black/60 dark:text-white/60 border-l-2 border-black/15 dark:border-white/20 pl-3">
                &ldquo;{claim.quote}&rdquo; &mdash; page {claim.page}
              </blockquote>
              <p className="text-xs text-black/40 dark:text-white/40">
                retrieval score {claim.retrievalScore.toFixed(3)}
              </p>
            </li>
          ))}
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Quiz ({quiz.length})</h2>
        {quiz.length === 0 && (
          <p className="text-sm text-black/50 dark:text-white/50">
            No quiz questions yet - generate some above.
          </p>
        )}
        <ul className="space-y-3">
          {quiz.map((q) => (
            <QuizItem key={q.id} question={q} />
          ))}
        </ul>
      </section>
    </div>
  );
}

function QuizItem({ question }: { question: QuizQuestion }) {
  const [revealed, setRevealed] = useState(false);
  return (
    <li className="rounded-lg border border-black/10 dark:border-white/15 p-4 space-y-2">
      <div className="flex items-start justify-between gap-3">
        <p>{question.question}</p>
        <GroundingBadge label={question.groundingLabel} score={question.groundingScore} />
      </div>
      <button
        onClick={() => setRevealed((r) => !r)}
        className="text-sm underline text-black/60 dark:text-white/60"
      >
        {revealed ? `Answer: ${question.answer} (page ${question.page})` : "Reveal answer"}
      </button>
    </li>
  );
}
