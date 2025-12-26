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
      this.config.embeddingModel,
      this.config.rerankerModel
    );
  }

  /** Find nearest cached query using cosine similarity (only queries with valid results) */
  async findNearestQuery(
    queryVec: number[],
    tenantId?: string
  ): Promise<CacheLookupResult | null> {
    const queriesCol = this.db.collection(COLLECTIONS.queries);
    const resultsCol = this.db.collection(COLLECTIONS.results);

    // Join queries with results to avoid orphaned query matches.
    // The subquery uses idx_query_id (persistent) for efficient lookup.
    // Vector index usage is unaffected by the join (verified via EXPLAIN).
    const searchQuery = tenantId
      ? aql`
          FOR q IN ${queriesCol}
            FILTER q.tenant_id == ${tenantId}
            FILTER q.q_vec != null
            LET r = FIRST(FOR res IN ${resultsCol} FILTER res.query_id == q._key RETURN res)
            FILTER r != null
            LET sim = COSINE_SIMILARITY(q.q_vec, ${queryVec})
            FILTER sim >= ${this.config.similarityThreshold}
            SORT sim DESC
            LIMIT 1
            RETURN { query: q, results: r, similarity: sim }
        `
      : aql`
          FOR q IN ${queriesCol}
            FILTER q.q_vec != null
            LET r = FIRST(FOR res IN ${resultsCol} FILTER res.query_id == q._key RETURN res)
            FILTER r != null
            LET sim = COSINE_SIMILARITY(q.q_vec, ${queryVec})
            FILTER sim >= ${this.config.similarityThreshold}
            SORT sim DESC
            LIMIT 1
            RETURN { query: q, results: r, similarity: sim }
        `;

    const cursor = await this.db.query(searchQuery);
    const match = await cursor.next();

    if (!match) return null;

    return {
      query: match.query as QueryDocument,
      results: match.results as QueryResultsDocument,
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

  /** Update cached results for an existing query (uses default TTL) */
  async updateResults(
    queryKey: string,
    items: CacheItem[]
  ): Promise<void> {
    await this.updateResultsWithIntent(queryKey, items);
  }

  /** Update cached results with proper intent-based TTL */
  async updateResultsWithIntent(
    queryKey: string,
    items: CacheItem[],
    intent?: ReturnType<typeof extractIntent>
  ): Promise<void> {
    const resultsCol = this.db.collection(COLLECTIONS.results);
    // Use provided intent or default (preserves volatile query TTL)
    const effectiveIntent = intent ?? { entities: [], facets: [], timebox: null };

    await this.db.query(aql`
      FOR r IN ${resultsCol}
        FILTER r.query_id == ${queryKey}
        UPDATE r WITH {
          items: ${items.slice(0, this.config.topKCached)},
          model_rev: ${this.modelRev},
          ttl_at: ${computeTtl(effectiveIntent, this.config.defaultTtlMs)},
          freshened_at: ${Date.now()}
        } IN ${resultsCol}
    `);
  }

  /**
   * Main retrieval method - check cache first, fallback to fresh retrieval
   * @param query - User query text
   * @param retrieveFn - Function to perform fresh retrieval if cache miss
   * @param tenantId - Optional tenant ID for multi-tenant caching (defaults to config.tenantId)
   */
  async retrieve(
    query: string,
    retrieveFn: (normalizedQuery: string, queryVec: number[]) => Promise<CacheItem[]>,
    tenantId?: string
  ): Promise<RetrievalResult> {
    // Use config tenantId as default if not provided
    const effectiveTenantId = tenantId ?? this.config.tenantId;

    // Normalize and embed the query
    const normalized = normalizeText(query);
    const queryVec = await embed(normalized, this.config.embeddingModel);
    const intent = extractIntent(query);

    // Check cache
    const cached = await this.findNearestQuery(queryVec, effectiveTenantId);

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
    const queryKey = await this.store(normalized, queryVec, freshItems, intent, effectiveTenantId);

    return {
      items: freshItems.slice(0, this.config.topKReturned),
      source: 'fresh',
      query_id: queryKey,
    };
  }

  /**
   * Async refresh pattern - return cached results immediately,
   * refresh in background if needed.
   *
   * Design note on similar-query refresh:
   * When a similar (but not identical) query matches an expired or stale cache entry,
   * we update the existing cached query's results rather than creating a new entry.
   * This is intentional:
   * - Similar queries are treated as semantically equivalent (above threshold)
   * - The original query's vector continues to match similar future queries
   * - Updating in-place prevents cache bloat from slight phrasings variations
   * - The retrieval function receives the NEW query's text/vector for fresh results
   * - Intent-based TTL is preserved from the original query for consistency
   */
  async retrieveWithBackgroundRefresh(
    query: string,
    retrieveFn: (normalizedQuery: string, queryVec: number[]) => Promise<CacheItem[]>,
    tenantId?: string
  ): Promise<RetrievalResult> {
    // Use config tenantId as default if not provided
    const effectiveTenantId = tenantId ?? this.config.tenantId;

    const normalized = normalizeText(query);
    const queryVec = await embed(normalized, this.config.embeddingModel);
    const intent = extractIntent(query);

    const cached = await this.findNearestQuery(queryVec, effectiveTenantId);

    if (cached && cached.results) {
      const isModelMatch = cached.results.model_rev === this.modelRev;
      const ttlTime = new Date(cached.results.ttl_at).getTime();
      const now = Date.now();
      const isExpired = ttlTime <= now;
      const isApproachingExpiry = ttlTime - now < TTL_PRESETS.realtime;

      // If expired, do NOT return stale data - treat as cache miss
      if (isExpired) {
        const freshItems = await retrieveFn(normalized, queryVec);
        // Update existing cache entry, preserving original intent for consistent TTL
        await this.updateResultsWithIntent(cached.query._key, freshItems, cached.query.intent);
        await this.touchQuery(cached.query._key);

        return {
          items: freshItems.slice(0, this.config.topKReturned),
          source: 'fresh',
          query_id: cached.query._key,
        };
      }

      // Not expired - return cached immediately
      await this.touchQuery(cached.query._key);

      // Check if refresh needed (model mismatch or approaching expiry)
      if (!isModelMatch || isApproachingExpiry) {
        // Background refresh - fire and forget, preserve original intent
        this.backgroundRefresh(cached.query._key, normalized, queryVec, retrieveFn, cached.query.intent);
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
    const queryKey = await this.store(normalized, queryVec, freshItems, intent, effectiveTenantId);

    return {
      items: freshItems.slice(0, this.config.topKReturned),
      source: 'fresh',
      query_id: queryKey,
    };
  }

  /** Background refresh helper - preserves original intent for correct TTL */
  private async backgroundRefresh(
    queryKey: string,
    normalized: string,
    queryVec: number[],
    retrieveFn: (normalizedQuery: string, queryVec: number[]) => Promise<CacheItem[]>,
    originalIntent?: ReturnType<typeof extractIntent>
  ): Promise<void> {
    try {
      const freshItems = await retrieveFn(normalized, queryVec);
      await this.updateResultsWithIntent(queryKey, freshItems, originalIntent);
    } catch (err) {
      console.error('Background refresh failed:', err);
    }
  }

  /** Get cache statistics (uses aggregation, does not load all docs) */
  async getStats(): Promise<{
    totalQueries: number;
    totalHits: number;
    hitRate: number;
  }> {
    const queriesCol = this.db.collection(COLLECTIONS.queries);

    // Use COLLECT for efficient aggregation without loading all docs
    const cursor = await this.db.query(aql`
      LET stats = (
        FOR q IN ${queriesCol}
          COLLECT AGGREGATE
            total = COUNT(1),
            hits = SUM(q.hit_count)
          RETURN { total, hits }
      )[0]
      RETURN {
        totalQueries: stats.total || 0,
        totalHits: stats.hits || 0,
        hitRate: (stats.total || 0) > 0 AND (stats.hits || 0) > 0
          ? ((stats.hits - stats.total) / stats.hits)
          : 0
      }
    `);

    const stats = await cursor.next();
    return stats || { totalQueries: 0, totalHits: 0, hitRate: 0 };
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
