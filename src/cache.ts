/**
 * Semantic Cache Core Logic
 */

import { Database, aql } from 'arangojs';
import { COLLECTIONS } from './db';
import {
  normalizeText,
  embed,
  extractIntent,
  getModelRevision,
  cosineSimilarity,
  type EmbeddingModel,
} from './embeddings';
import {
  type SemanticCacheConfig,
  type CacheItem,
  type QueryDocument,
  type QueryResultsDocument,
  type CacheLookupResult,
  type RetrievalResult,
  DEFAULT_CONFIG,
  TTL_PRESETS,
} from './types';

/** Generate a ULID-like key */
function generateKey(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 10);
  return `${timestamp}${random}`.toUpperCase();
}

/** Calculate TTL based on intent volatility */
function computeTtl(intent: ReturnType<typeof extractIntent>, baseTtlMs: number): string {
  let ttlMs = baseTtlMs;

  // Adjust TTL based on timebox (time-sensitive queries expire faster)
  if (intent.timebox) {
    ttlMs = TTL_PRESETS.dynamic;
  }

  // Adjust based on facets
  const volatileFacets = ['price', 'review', 'today', 'news'];
  if (intent.facets.some((f) => volatileFacets.includes(f))) {
    ttlMs = Math.min(ttlMs, TTL_PRESETS.dynamic);
  }

  const ttlAt = new Date(Date.now() + ttlMs);
  return ttlAt.toISOString();
}

export class SemanticCache {
  private db: Database;
  private config: SemanticCacheConfig;
  private modelRev: string;

  constructor(db: Database, config: Partial<SemanticCacheConfig> = {}) {
    this.db = db;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.modelRev = getModelRevision(
      this.config.embeddingModel as EmbeddingModel,
      this.config.rerankerModel
    );
  }

  /** Find nearest cached query using cosine similarity */
  async findNearestQuery(
    queryVec: number[],
    tenantId?: string
  ): Promise<CacheLookupResult | null> {
    const queriesCol = this.db.collection(COLLECTIONS.queries);
    const resultsCol = this.db.collection(COLLECTIONS.results);

    // nProbe controls search quality vs speed (set via index defaultNProbe)
    // ArangoDB 3.12.4+ with vector index automatically uses indexed search
    const searchQuery = tenantId
      ? aql`
          FOR q IN ${queriesCol}
            FILTER q.tenant_id == ${tenantId}
            FILTER q.q_vec != null
            LET sim = COSINE_SIMILARITY(q.q_vec, ${queryVec})
            FILTER sim >= ${this.config.similarityThreshold}
            SORT sim DESC
            LIMIT 1
            RETURN { query: q, similarity: sim }
        `
      : aql`
          FOR q IN ${queriesCol}
            FILTER q.q_vec != null
            LET sim = COSINE_SIMILARITY(q.q_vec, ${queryVec})
            FILTER sim >= ${this.config.similarityThreshold}
            SORT sim DESC
            LIMIT 1
            RETURN { query: q, similarity: sim }
        `;

    const cursor = await this.db.query(searchQuery);
    const match = await cursor.next();

    if (!match) return null;

    // Fetch associated results
    const resultsCursor = await this.db.query(aql`
      FOR r IN ${resultsCol}
        FILTER r.query_id == ${match.query._key}
        LIMIT 1
        RETURN r
    `);

    const results = await resultsCursor.next();

    return {
      query: match.query as QueryDocument,
      results: results as QueryResultsDocument | null,
      similarity: match.similarity,
    };
  }

  /** Update hit count and last_hit_at for a cached query */
  async touchQuery(queryKey: string): Promise<void> {
    const queriesCol = this.db.collection(COLLECTIONS.queries);

    await this.db.query(aql`
      FOR q IN ${queriesCol}
        FILTER q._key == ${queryKey}
        UPDATE q WITH {
          last_hit_at: ${Date.now()},
          hit_count: q.hit_count + 1
        } IN ${queriesCol}
    `);
  }

  /** Store a new query and its results in the cache */
  async store(
    normalizedText: string,
    queryVec: number[],
    items: CacheItem[],
    intent: ReturnType<typeof extractIntent>,
    tenantId?: string
  ): Promise<string> {
    const queriesCol = this.db.collection(COLLECTIONS.queries);
    const resultsCol = this.db.collection(COLLECTIONS.results);

    const queryKey = generateKey();
    const now = Date.now();
    const ttlAt = computeTtl(intent, this.config.defaultTtlMs);

    // Store query document
    const queryDoc: Omit<QueryDocument, '_id' | '_rev'> = {
      _key: queryKey,
      q_text_norm: normalizedText,
      q_vec: queryVec,
      intent,
      created_at: now,
      last_hit_at: now,
      hit_count: 1,
      ...(tenantId && { tenant_id: tenantId }),
    };

    await queriesCol.save(queryDoc);

    // Store results document
    const resultsDoc: Omit<QueryResultsDocument, '_id' | '_rev' | '_key'> = {
      query_id: queryKey,
      items: items.slice(0, this.config.topKCached),
      model_rev: this.modelRev,
      ttl_at: ttlAt,
    };

    await resultsCol.save(resultsDoc);

    return queryKey;
  }

  /** Update cached results for an existing query */
  async updateResults(
    queryKey: string,
    items: CacheItem[]
  ): Promise<void> {
    const resultsCol = this.db.collection(COLLECTIONS.results);
    const intent = extractIntent(''); // Default intent for TTL calculation

    await this.db.query(aql`
      FOR r IN ${resultsCol}
        FILTER r.query_id == ${queryKey}
        UPDATE r WITH {
          items: ${items.slice(0, this.config.topKCached)},
          model_rev: ${this.modelRev},
          ttl_at: ${computeTtl(intent, this.config.defaultTtlMs)},
          freshened_at: ${Date.now()}
        } IN ${resultsCol}
    `);
  }

  /**
   * Main retrieval method - check cache first, fallback to fresh retrieval
   * @param query - User query text
   * @param retrieveFn - Function to perform fresh retrieval if cache miss
   * @param tenantId - Optional tenant ID for multi-tenant caching
   */
  async retrieve(
    query: string,
    retrieveFn: (normalizedQuery: string, queryVec: number[]) => Promise<CacheItem[]>,
    tenantId?: string
  ): Promise<RetrievalResult> {
    // Normalize and embed the query
    const normalized = normalizeText(query);
    const queryVec = await embed(normalized, this.config.embeddingModel as EmbeddingModel);
    const intent = extractIntent(query);

    // Check cache
    const cached = await this.findNearestQuery(queryVec, tenantId);

    if (cached && cached.results) {
      // Validate cache entry
      const isModelMatch = cached.results.model_rev === this.modelRev;
      const isNotExpired = new Date(cached.results.ttl_at) > new Date();

      if (isModelMatch && isNotExpired) {
        // Cache hit - update stats and return
        await this.touchQuery(cached.query._key);

        return {
          items: cached.results.items.slice(0, this.config.topKReturned),
          source: 'semantic-cache',
          similarity: cached.similarity,
          query_id: cached.query._key,
        };
      }
    }

    // Cache miss - perform fresh retrieval
    const freshItems = await retrieveFn(normalized, queryVec);

    // Store in cache
    const queryKey = await this.store(normalized, queryVec, freshItems, intent, tenantId);

    return {
      items: freshItems.slice(0, this.config.topKReturned),
      source: 'fresh',
      query_id: queryKey,
    };
  }

  /**
   * Async refresh pattern - return cached results immediately,
   * refresh in background if needed
   */
  async retrieveWithBackgroundRefresh(
    query: string,
    retrieveFn: (normalizedQuery: string, queryVec: number[]) => Promise<CacheItem[]>,
    tenantId?: string
  ): Promise<RetrievalResult> {
    const normalized = normalizeText(query);
    const queryVec = await embed(normalized, this.config.embeddingModel as EmbeddingModel);
    const intent = extractIntent(query);

    const cached = await this.findNearestQuery(queryVec, tenantId);

    if (cached && cached.results) {
      // Return cached immediately
      await this.touchQuery(cached.query._key);

      // Check if refresh needed (model mismatch or approaching expiry)
      const needsRefresh =
        cached.results.model_rev !== this.modelRev ||
        new Date(cached.results.ttl_at).getTime() - Date.now() < TTL_PRESETS.realtime;

      if (needsRefresh) {
        // Background refresh - fire and forget
        this.backgroundRefresh(cached.query._key, normalized, queryVec, retrieveFn);
      }

      return {
        items: cached.results.items.slice(0, this.config.topKReturned),
        source: 'semantic-cache',
        similarity: cached.similarity,
        query_id: cached.query._key,
      };
    }

    // No cache - fresh retrieval
    const freshItems = await retrieveFn(normalized, queryVec);
    const queryKey = await this.store(normalized, queryVec, freshItems, intent, tenantId);

    return {
      items: freshItems.slice(0, this.config.topKReturned),
      source: 'fresh',
      query_id: queryKey,
    };
  }

  /** Background refresh helper */
  private async backgroundRefresh(
    queryKey: string,
    normalized: string,
    queryVec: number[],
    retrieveFn: (normalizedQuery: string, queryVec: number[]) => Promise<CacheItem[]>
  ): Promise<void> {
    try {
      const freshItems = await retrieveFn(normalized, queryVec);
      await this.updateResults(queryKey, freshItems);
    } catch (err) {
      console.error('Background refresh failed:', err);
    }
  }

  /** Get cache statistics */
  async getStats(): Promise<{
    totalQueries: number;
    totalHits: number;
    hitRate: number;
    avgSimilarity: number;
  }> {
    const queriesCol = this.db.collection(COLLECTIONS.queries);

    const cursor = await this.db.query(aql`
      LET queries = (FOR q IN ${queriesCol} RETURN q)
      LET total = LENGTH(queries)
      LET hits = SUM(FOR q IN queries RETURN q.hit_count)
      RETURN {
        totalQueries: total,
        totalHits: hits,
        hitRate: total > 0 ? (hits - total) / hits : 0
      }
    `);

    const stats = await cursor.next();
    return stats || { totalQueries: 0, totalHits: 0, hitRate: 0, avgSimilarity: 0 };
  }

  /** Invalidate cache entries by model revision */
  async invalidateByModelRev(modelRev: string): Promise<number> {
    const resultsCol = this.db.collection(COLLECTIONS.results);

    const cursor = await this.db.query(aql`
      FOR r IN ${resultsCol}
        FILTER r.model_rev == ${modelRev}
        REMOVE r IN ${resultsCol}
        RETURN OLD._key
    `);

    const removed = await cursor.all();
    return removed.length;
  }

  /** Clear all cache entries */
  async clear(): Promise<void> {
    const queriesCol = this.db.collection(COLLECTIONS.queries);
    const resultsCol = this.db.collection(COLLECTIONS.results);

    await this.db.query(aql`FOR r IN ${resultsCol} REMOVE r IN ${resultsCol}`);
    await this.db.query(aql`FOR q IN ${queriesCol} REMOVE q IN ${queriesCol}`);
  }
}
