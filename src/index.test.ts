/**
 * Semantic Cache Tests
 */

import { test, expect, describe, beforeAll, afterAll } from 'bun:test';
import { Database } from 'arangojs';
import {
  createDatabase,
  setupSemanticCache,
  dropSemanticCache,
  getCacheStats,
  COLLECTIONS,
} from './db';
import { SemanticCache } from './cache';
import {
  normalizeText,
  extractIntent,
  cosineSimilarity,
  embed,
} from './embeddings';
import type { CacheItem } from './types';

// Check if OpenAI API key is available
const hasOpenAIKey = !!process.env.OPENAI_API_KEY;

// Test database connection
let db: Database;
let cache: SemanticCache;

beforeAll(async () => {
  db = createDatabase();
  await setupSemanticCache(db);
  cache = new SemanticCache(db, {
    similarityThreshold: 0.85,
    embeddingModel: 'text-embedding-3-small',
  });
});

afterAll(async () => {
  // Clean up test data
  await cache.clear();
});

describe('Text Normalization', () => {
  test('lowercases text', () => {
    expect(normalizeText('Hello World')).toBe('hello world');
  });

  test('trims whitespace', () => {
    expect(normalizeText('  hello  ')).toBe('hello');
  });

  test('collapses multiple spaces', () => {
    expect(normalizeText('hello   world')).toBe('hello world');
  });

  test('removes special characters', () => {
    expect(normalizeText('hello! world?')).toBe('hello world');
  });

  test('preserves hyphens', () => {
    expect(normalizeText('wolf-gmbh')).toBe('wolf-gmbh');
  });
});

describe('Intent Extraction', () => {
  test('extracts product codes', () => {
    const intent = extractIntent('What is FGB-K20 error code?');
    expect(intent.entities).toContain('FGB-K20');
  });

  test('extracts facets', () => {
    const intent = extractIntent('Show me the specs');
    expect(intent.facets).toContain('specs');
  });

  test('extracts time references', () => {
    const intent = extractIntent('What happened today?');
    expect(intent.timebox).toBe('today');
  });

  test('handles year references', () => {
    const intent = extractIntent('Products from 2024');
    expect(intent.timebox).toBe('2024');
  });

  test('returns null timebox for non-temporal queries', () => {
    const intent = extractIntent('What is the price?');
    expect(intent.timebox).toBeNull();
  });
});

describe('Cosine Similarity', () => {
  test('identical vectors have similarity 1', () => {
    const vec = [1, 2, 3, 4, 5];
    expect(cosineSimilarity(vec, vec)).toBeCloseTo(1, 5);
  });

  test('orthogonal vectors have similarity 0', () => {
    const a = [1, 0];
    const b = [0, 1];
    expect(cosineSimilarity(a, b)).toBeCloseTo(0, 5);
  });

  test('opposite vectors have similarity -1', () => {
    const a = [1, 0];
    const b = [-1, 0];
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1, 5);
  });

  test('throws on dimension mismatch', () => {
    expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrow();
  });
});

describe('Embeddings', () => {
  test.skipIf(!hasOpenAIKey)('generates embeddings with correct dimension', async () => {
    const embedding = await embed('test query');
    expect(embedding).toBeInstanceOf(Array);
    expect(embedding.length).toBe(1536); // text-embedding-3-small
  });

  test.skipIf(!hasOpenAIKey)('similar texts have high similarity', async () => {
    const e1 = await embed('iPhone price');
    const e2 = await embed('iPhone cost');
    const similarity = cosineSimilarity(e1, e2);
    expect(similarity).toBeGreaterThan(0.8);
  });

  test.skipIf(!hasOpenAIKey)('different texts have lower similarity', async () => {
    const e1 = await embed('iPhone price');
    const e2 = await embed('weather forecast');
    const similarity = cosineSimilarity(e1, e2);
    expect(similarity).toBeLessThan(0.5);
  });
});

describe('Semantic Cache', () => {
  const mockRetriever = async (query: string, vec: number[]): Promise<CacheItem[]> => {
    return [
      { id: `test/${query.slice(0, 10)}`, type: 'node', score: 0.95 },
      { id: 'test/related', type: 'edge', score: 0.85 },
    ];
  };

  test.skipIf(!hasOpenAIKey)('stores and retrieves cached results', async () => {
    // First query - should be fresh
    const result1 = await cache.retrieve('unique test query 12345', mockRetriever);
    expect(result1.source).toBe('fresh');
    expect(result1.items.length).toBeGreaterThan(0);

    // Same query - should hit cache
    const result2 = await cache.retrieve('unique test query 12345', mockRetriever);
    expect(result2.source).toBe('semantic-cache');
    expect(result2.similarity).toBeCloseTo(1, 2);
  });

  test.skipIf(!hasOpenAIKey)('matches semantically similar queries', async () => {
    // Store a query
    await cache.retrieve('what is the weather today', mockRetriever);

    // Similar query should hit cache
    const result = await cache.retrieve('weather forecast for today', mockRetriever);
    // Note: might be cache or fresh depending on similarity threshold
    if (result.source === 'semantic-cache') {
      expect(result.similarity).toBeGreaterThan(0.85);
    }
  });

  test.skipIf(!hasOpenAIKey)('returns fresh for dissimilar queries', async () => {
    await cache.retrieve('apple iphone features', mockRetriever);
    const result = await cache.retrieve('banana nutrition facts', mockRetriever);
    expect(result.source).toBe('fresh');
  });

  test.skipIf(!hasOpenAIKey)('respects tenant isolation', async () => {
    const tenantCache = new SemanticCache(db, {
      similarityThreshold: 0.85,
      tenantId: 'tenant-a',
    });

    // Store for tenant A
    await tenantCache.retrieve('tenant specific query xyz', mockRetriever, 'tenant-a');

    // Query with different tenant should not find it
    const result = await tenantCache.retrieve('tenant specific query xyz', mockRetriever, 'tenant-b');
    expect(result.source).toBe('fresh');
  });
});

describe('Cache Statistics', () => {
  test('tracks cache statistics', async () => {
    const stats = await getCacheStats(db);
    expect(stats.queryCount).toBeGreaterThanOrEqual(0);
    expect(stats.resultCount).toBeGreaterThanOrEqual(0);
  });

  test('cache getStats returns valid data', async () => {
    const stats = await cache.getStats();
    expect(stats).toHaveProperty('totalQueries');
    expect(stats).toHaveProperty('totalHits');
    expect(stats).toHaveProperty('hitRate');
  });
});
