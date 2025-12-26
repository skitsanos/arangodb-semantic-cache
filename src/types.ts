/**
 * Semantic Cache Types for Graph-RAG Pipeline
 */

/** Supported embedding models */
export type EmbeddingModelType =
  | 'text-embedding-3-small'
  | 'text-embedding-3-large'
  | 'text-embedding-ada-002';

/** Intent extracted from a query for analytics and fine-tuning */
export interface QueryIntent {
  entities: string[];
  facets: string[];
  timebox: string | null;
}

/** Cached result item - a node or edge from the knowledge graph */
export interface CacheItem {
  id: string;
  type: 'node' | 'edge';
  score: number;
}

/** Stored query document in ArangoDB */
export interface QueryDocument {
  _key: string;
  _id?: string;
  _rev?: string;
  q_text_norm: string;
  q_vec: number[];
  intent: QueryIntent;
  created_at: number;
  last_hit_at: number;
  hit_count: number;
  tenant_id?: string;
  lang?: string;
}

/** Cached results document in ArangoDB */
export interface QueryResultsDocument {
  _key?: string;
  _id?: string;
  _rev?: string;
  query_id: string;
  items: CacheItem[];
  model_rev: string;
  ttl_at: string; // ISO date string for TTL index
  freshened_at?: number;
}

/** Cache lookup result */
export interface CacheLookupResult {
  query: QueryDocument;
  results: QueryResultsDocument | null;
  similarity: number;
}

/** Retrieval result from the semantic cache */
export interface RetrievalResult {
  items: CacheItem[];
  source: 'semantic-cache' | 'fresh';
  similarity?: number;
  query_id?: string;
}

/** Configuration for the semantic cache */
export interface SemanticCacheConfig {
  /** Cosine similarity threshold (0.83-0.88 recommended) */
  similarityThreshold: number;
  /** Embedding model identifier */
  embeddingModel: EmbeddingModelType;
  /** Reranker model identifier (optional) */
  rerankerModel?: string;
  /** Default TTL in milliseconds */
  defaultTtlMs: number;
  /** Number of results to cache */
  topKCached: number;
  /** Number of results to return */
  topKReturned: number;
  /** Vector dimension (must match embedding model) */
  vectorDimension: number;
  /** Optional tenant ID for multi-tenant caching */
  tenantId?: string;
}

/** Default configuration values */
export const DEFAULT_CONFIG: SemanticCacheConfig = {
  similarityThreshold: 0.85,
  embeddingModel: 'text-embedding-3-small',
  defaultTtlMs: 7 * 24 * 60 * 60 * 1000, // 7 days
  topKCached: 25,
  topKReturned: 10,
  vectorDimension: 1536, // text-embedding-3-small dimension
};

/** TTL presets based on content volatility */
export const TTL_PRESETS = {
  static: 30 * 24 * 60 * 60 * 1000,      // 30 days
  semiDynamic: 7 * 24 * 60 * 60 * 1000,  // 7 days
  dynamic: 24 * 60 * 60 * 1000,           // 24 hours
  realtime: 2 * 60 * 60 * 1000,           // 2 hours
} as const;
