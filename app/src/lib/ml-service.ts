import {
  GroundedClaimSchema,
  NarrativeTemplateSchema,
  ParsedDocumentSchema,
  PublicationBundleSchema,
  PublicationSchema,
  QuizQuestionSchema,
  RelatedDocumentSchema,
  SectionHealthSchema,
  type GroundedClaim,
  type NarrativeTemplate,
  type ParsedDocument,
  type Publication,
  type PublicationBundle,
  type QuizQuestion,
  type RelatedDocument,
  type SectionHealth,
} from "./schemas/document";
import { z } from "zod";

export const ML_SERVICE_URL = process.env.NEXT_PUBLIC_ML_SERVICE_URL ?? "http://localhost:8000";

async function request<T>(
  path: string,
  schema: z.ZodType<T>,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${ML_SERVICE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`${init?.method ?? "GET"} ${path} failed (${response.status}): ${detail}`);
  }
  return schema.parse(await response.json());
}

export function ingestArxivPaper(arxivId: string): Promise<ParsedDocument> {
  return request("/documents/arxiv", ParsedDocumentSchema, {
    method: "POST",
    body: JSON.stringify({ arxiv_id: arxivId }),
  });
}

export function getDocument(documentId: string): Promise<ParsedDocument> {
  return request(`/documents/${documentId}`, ParsedDocumentSchema);
}

export function listClaims(documentId: string): Promise<GroundedClaim[]> {
  return request(`/documents/${documentId}/claims`, z.array(GroundedClaimSchema));
}

export function generateClaims(
  documentId: string,
  params: { query: string; section?: string; k?: number; maxClaims?: number },
): Promise<GroundedClaim[]> {
  return request(`/documents/${documentId}/claims/generate`, z.array(GroundedClaimSchema), {
    method: "POST",
    body: JSON.stringify({
      query: params.query,
      section: params.section ?? null,
      k: params.k ?? 5,
      max_claims: params.maxClaims ?? 5,
    }),
  });
}

export function regenerateClaims(
  documentId: string,
  params: { query: string; section: string | null; k?: number; maxClaims?: number },
): Promise<GroundedClaim[]> {
  return request(`/documents/${documentId}/claims/regenerate`, z.array(GroundedClaimSchema), {
    method: "POST",
    body: JSON.stringify({
      query: params.query,
      section: params.section,
      k: params.k ?? 5,
      max_claims: params.maxClaims ?? 5,
    }),
  });
}

export function getDocumentHealth(documentId: string): Promise<SectionHealth[]> {
  return request(`/documents/${documentId}/health`, z.array(SectionHealthSchema));
}

export function listTemplates(): Promise<NarrativeTemplate[]> {
  return request("/templates", z.array(NarrativeTemplateSchema));
}

export function getRelatedDocuments(documentId: string, k = 5): Promise<RelatedDocument[]> {
  return request(
    `/documents/${documentId}/related?k=${k}`,
    z.array(RelatedDocumentSchema),
  );
}

export function publishDocument(
  documentId: string,
  params: { expiresInHours?: number | null; includeFigures?: boolean },
): Promise<Publication> {
  return request(`/documents/${documentId}/publish`, PublicationSchema, {
    method: "POST",
    body: JSON.stringify({
      expires_in_hours: params.expiresInHours ?? null,
      include_figures: params.includeFigures ?? true,
    }),
  });
}

export function getPublication(publicationId: string): Promise<PublicationBundle> {
  return request(`/publications/${publicationId}`, PublicationBundleSchema);
}

export function exportUrl(publicationId: string): string {
  return `${ML_SERVICE_URL}/publications/${publicationId}/export`;
}

export function listQuiz(documentId: string): Promise<QuizQuestion[]> {
  return request(`/documents/${documentId}/quiz`, z.array(QuizQuestionSchema));
}

export function generateQuiz(
  documentId: string,
  params: { query: string; section?: string; k?: number; maxQuestions?: number },
): Promise<QuizQuestion[]> {
  return request(`/documents/${documentId}/quiz/generate`, z.array(QuizQuestionSchema), {
    method: "POST",
    body: JSON.stringify({
      query: params.query,
      section: params.section ?? null,
      k: params.k ?? 5,
      max_questions: params.maxQuestions ?? 5,
    }),
  });
}
