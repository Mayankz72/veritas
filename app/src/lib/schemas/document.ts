import { z } from "zod";

/**
 * Mirrors ml-service/app/schemas/document.py and
 * packages/shared/README.md - kept in sync by hand (TS/Python don't share
 * a runtime). Update all three together when a shape changes.
 */

export const ChunkSchema = z.object({
  id: z.string(),
  page: z.number().int(),
  section: z.string().nullable(),
  text: z.string(),
  bbox: z.tuple([z.number(), z.number(), z.number(), z.number()]).nullable(),
});
export type Chunk = z.infer<typeof ChunkSchema>;

export const ParsedDocumentSchema = z.object({
  id: z.string(),
  sourceType: z.enum(["upload", "arxiv"]),
  title: z.string(),
  pageCount: z.number().int(),
  chunks: z.array(ChunkSchema),
});
export type ParsedDocument = z.infer<typeof ParsedDocumentSchema>;

export const GroundingLabelSchema = z.enum(["supported", "unsupported", "partial"]);
export type GroundingLabel = z.infer<typeof GroundingLabelSchema>;

export const GroundedClaimSchema = z.object({
  id: z.string(),
  documentId: z.string(),
  section: z.string().nullable(),
  text: z.string(),
  sourceChunkIds: z.array(z.string()),
  page: z.number().int(),
  quote: z.string(),
  retrievalScore: z.number(),
  groundingLabel: GroundingLabelSchema.nullable(),
  groundingScore: z.number().nullable(),
  version: z.number().int(),
  isCurrent: z.boolean(),
});
export type GroundedClaim = z.infer<typeof GroundedClaimSchema>;

export const NarrativeTemplateSchema = z.object({
  key: z.string(),
  label: z.string(),
  query: z.string(),
});
export type NarrativeTemplate = z.infer<typeof NarrativeTemplateSchema>;

export const QuizQuestionSchema = z.object({
  id: z.string(),
  documentId: z.string(),
  section: z.string().nullable(),
  question: z.string(),
  answer: z.string(),
  sourceChunkId: z.string(),
  page: z.number().int(),
  groundingLabel: GroundingLabelSchema.nullable(),
  groundingScore: z.number().nullable(),
});
export type QuizQuestion = z.infer<typeof QuizQuestionSchema>;

export const PublicationSchema = z.object({
  id: z.string(),
  documentId: z.string(),
  includeFigures: z.boolean(),
  expiresAt: z.string().nullable(),
  createdAt: z.string(),
});
export type Publication = z.infer<typeof PublicationSchema>;

export const PublicationBundleSchema = z.object({
  publication: PublicationSchema,
  document: ParsedDocumentSchema,
  claims: z.array(GroundedClaimSchema),
  quiz: z.array(QuizQuestionSchema),
});
export type PublicationBundle = z.infer<typeof PublicationBundleSchema>;

export const SectionHealthSchema = z.object({
  section: z.string().nullable(),
  claimCount: z.number().int(),
  supportedCount: z.number().int(),
  partialCount: z.number().int(),
  unsupportedCount: z.number().int(),
  notCheckedCount: z.number().int(),
  averageGroundingScore: z.number().nullable(),
  flagged: z.boolean(),
  flagReason: z.string().nullable(),
});
export type SectionHealth = z.infer<typeof SectionHealthSchema>;
