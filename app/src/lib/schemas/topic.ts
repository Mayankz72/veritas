import { z } from "zod";
import { GroundingLabelSchema } from "./document";

/** Mirrors ml-service/app/schemas/topic.py - keep the two in sync by hand. */

export const ArxivCandidateSchema = z.object({
  arxivId: z.string(),
  title: z.string(),
  authors: z.array(z.string()),
  published: z.string(),
  abstract: z.string(),
  score: z.number(),
  isRecent: z.boolean(),
});
export type ArxivCandidate = z.infer<typeof ArxivCandidateSchema>;

export const BriefItemSchema = z.object({
  id: z.string(),
  text: z.string(),
  quote: z.string(),
  page: z.number().int(),
  retrievalScore: z.number(),
  groundingLabel: GroundingLabelSchema.nullable(),
  groundingScore: z.number().nullable(),
});
export type BriefItem = z.infer<typeof BriefItemSchema>;

export const PaperBriefSchema = z.object({
  problem: z.array(BriefItemSchema),
  method: z.array(BriefItemSchema),
  results: z.array(BriefItemSchema),
  matters: z.array(BriefItemSchema),
});
export type PaperBrief = z.infer<typeof PaperBriefSchema>;

export const PaperStatusSchema = z.enum(["pending", "ingesting", "analyzing", "done", "error"]);
export type PaperStatus = z.infer<typeof PaperStatusSchema>;

export const TopicPaperSchema = z.object({
  id: z.string(),
  arxivId: z.string(),
  title: z.string(),
  authors: z.array(z.string()),
  published: z.string(),
  abstract: z.string(),
  status: PaperStatusSchema,
  error: z.string().nullable(),
  documentId: z.string().nullable(),
  briefMode: z.string().nullable(),
  brief: PaperBriefSchema,
});
export type TopicPaper = z.infer<typeof TopicPaperSchema>;

export const RelationshipSchema = z.object({
  fromPaperId: z.string(),
  toPaperId: z.string(),
  kind: z.enum(["cites", "similar"]),
  explanation: z.string(),
  page: z.number().int().nullable(),
  snippet: z.string().nullable(),
  sharedTerms: z.array(z.string()),
  similarity: z.number().nullable(),
});
export type Relationship = z.infer<typeof RelationshipSchema>;

export const TopicSynthesisSchema = z.object({
  mode: z.string(),
  overview: z.string(),
  readingOrder: z.array(z.string()),
  timeline: z.array(
    z.object({
      paperId: z.string(),
      label: z.string(),
      title: z.string(),
      year: z.string(),
      published: z.string(),
    }),
  ),
  relationships: z.array(RelationshipSchema),
  howTheyFit: z.array(z.object({ paperIds: z.array(z.string()), text: z.string() })),
  comparison: z.array(
    z.object({
      paperId: z.string(),
      label: z.string(),
      title: z.string(),
      year: z.string(),
      problem: z.string(),
      method: z.string(),
      results: z.string(),
      matters: z.string(),
    }),
  ),
  foundational: z.array(z.string()),
});
export type TopicSynthesis = z.infer<typeof TopicSynthesisSchema>;

export const TopicStatusSchema = z.enum([
  "searching",
  "ingesting",
  "analyzing",
  "synthesizing",
  "done",
  "error",
]);
export type TopicStatus = z.infer<typeof TopicStatusSchema>;

export const TopicSchema = z.object({
  id: z.string(),
  query: z.string(),
  status: TopicStatusSchema,
  error: z.string().nullable(),
  createdAt: z.string(),
  papers: z.array(TopicPaperSchema),
  synthesis: TopicSynthesisSchema.nullable(),
});
export type Topic = z.infer<typeof TopicSchema>;

export const TopicSummarySchema = z.object({
  id: z.string(),
  query: z.string(),
  status: z.string(),
  paperCount: z.number().int(),
  createdAt: z.string(),
});
export type TopicSummary = z.infer<typeof TopicSummarySchema>;
