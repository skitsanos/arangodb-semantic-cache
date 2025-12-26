/**
 * OpenAI Embeddings Integration
 */

import OpenAI from 'openai';
import type { QueryIntent, EmbeddingModelType } from './types';

const openai = new OpenAI();

/** Embedding model configurations */
export const EMBEDDING_MODELS: Record<EmbeddingModelType, { dimension: number }> = {
  'text-embedding-3-small': { dimension: 1536 },
  'text-embedding-3-large': { dimension: 3072 },
  'text-embedding-ada-002': { dimension: 1536 },
};

/** @deprecated Use EmbeddingModelType from types.ts */
export type EmbeddingModel = EmbeddingModelType;

/** Normalize query text for consistent caching */
export function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, ' ') // remove special chars except hyphens
    .replace(/\s+/g, ' ')      // collapse whitespace
    .trim();
}

/** Generate embedding for text using OpenAI */
export async function embed(
  text: string,
  model: EmbeddingModelType = 'text-embedding-3-small'
): Promise<number[]> {
  const response = await openai.embeddings.create({
    model,
    input: text,
    encoding_format: 'float',
  });

  return response.data[0].embedding;
}

/** Batch embed multiple texts */
export async function embedBatch(
  texts: string[],
  model: EmbeddingModelType = 'text-embedding-3-small'
): Promise<number[][]> {
  if (texts.length === 0) return [];

  const response = await openai.embeddings.create({
    model,
    input: texts,
    encoding_format: 'float',
  });

  // Sort by index to maintain order
  return response.data
    .sort((a, b) => a.index - b.index)
    .map((d) => d.embedding);
}

/** Calculate cosine similarity between two vectors */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimension mismatch: ${a.length} vs ${b.length}`);
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/** Simple intent extraction from query text */
export function extractIntent(text: string): QueryIntent {
  const normalized = normalizeText(text);
  const words = normalized.split(' ');

  // Extract potential entities (capitalized words, product codes, etc.)
  const entities: string[] = [];
  const facets: string[] = [];

  // Common facet keywords
  const facetKeywords = [
    'error', 'code', 'specs', 'specification', 'price', 'review',
    'manual', 'guide', 'install', 'setup', 'compare', 'vs',
    'how', 'what', 'where', 'when', 'why', 'which',
  ];

  // Detect product codes (alphanumeric patterns)
  const productCodePattern = /\b[a-z]{2,4}[-_]?[a-z0-9]{2,8}\b/gi;
  const codes = text.match(productCodePattern) || [];
  entities.push(...codes.map((c) => c.toUpperCase()));

  // Extract facets from keywords
  for (const word of words) {
    if (facetKeywords.includes(word)) {
      facets.push(word);
    }
  }

  // Detect time-related queries
  let timebox: string | null = null;
  const timePatterns = [
    /\b(today|yesterday|this week|last week|this month|last month)\b/i,
    /\b(20\d{2})\b/, // year
    /\b(q[1-4])\b/i, // quarter
  ];

  for (const pattern of timePatterns) {
    const match = text.match(pattern);
    if (match) {
      timebox = match[1];
      break;
    }
  }

  return {
    entities: [...new Set(entities)],
    facets: [...new Set(facets)],
    timebox,
  };
}

/** Generate model revision string for cache invalidation */
export function getModelRevision(
  embeddingModel: EmbeddingModelType,
  rerankerModel?: string
): string {
  const parts = [`embed:${embeddingModel}`];
  if (rerankerModel) {
    parts.push(`rerank:${rerankerModel}`);
  }
  return parts.join('|');
}
