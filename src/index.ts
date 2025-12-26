/**
 * Semantic Cache for Graph-RAG Pipelines
 *
 * A high-performance semantic caching layer that stores query embeddings
 * and their results, enabling instant responses for similar queries.
 */

// Export all public APIs
export { SemanticCache } from './cache';
export {
  createDatabase,
  setupSemanticCache,
  dropSemanticCache,
  getCacheStats,
  evictOldEntries,
  cleanupOrphanedQueries,
  createVectorIndex,
  hasVectorIndex,
  COLLECTIONS,
} from './db';
export {
  embed,
  embedBatch,
  normalizeText,
  extractIntent,
  cosineSimilarity,
  getModelRevision,
  EMBEDDING_MODELS,
  type EmbeddingModel,
} from './embeddings';
export {
  setupKnowledgeGraph,
  seedKnowledgeGraph,
  graphRAGRetrieval,
  dropKnowledgeGraph,
  GRAPH_COLLECTIONS,
  GRAPH_NAME,
} from './graph-rag';
export * from './types';
export type { EmbeddingModelType } from './types';

// Demo / CLI usage
import { createDatabase, setupSemanticCache, getCacheStats } from './db';
import { SemanticCache } from './cache';
import type { CacheItem } from './types';

async function demo() {
  console.log('🚀 Semantic Cache Demo\n');

  // Initialize database
  const db = createDatabase();
  console.log('Connected to ArangoDB');

  // Setup collections and indexes
  await setupSemanticCache(db);

  // Create cache instance
  const cache = new SemanticCache(db, {
    similarityThreshold: 0.85,
    embeddingModel: 'text-embedding-3-small',
    defaultTtlMs: 7 * 24 * 60 * 60 * 1000, // 7 days
  });

  // Mock Graph-RAG retrieval function
  const mockGraphRAG = async (query: string, queryVec: number[]): Promise<CacheItem[]> => {
    console.log(`  📊 Performing fresh Graph-RAG retrieval for: "${query}"`);
    // Simulate retrieval delay
    await new Promise((r) => setTimeout(r, 500));

    // Return mock results
    return [
      { id: 'products/apple-iphone-15', type: 'node', score: 0.95 },
      { id: 'products/apple-iphone-14', type: 'node', score: 0.88 },
      { id: 'categories/smartphones', type: 'node', score: 0.82 },
      { id: 'relations/compatible-with', type: 'edge', score: 0.79 },
    ];
  };

  // Test queries - first should be fresh, second should hit cache
  const queries = [
    'What is the price of iPhone 15?',
    'iPhone 15 price',  // Similar query - should hit cache
    'How much does iPhone 15 cost?',  // Another paraphrase
    'Samsung Galaxy S24 specs',  // Different query - should be fresh
  ];

  console.log('\n📝 Running test queries:\n');

  for (const query of queries) {
    console.log(`Query: "${query}"`);
    const start = performance.now();

    const result = await cache.retrieve(query, mockGraphRAG);

    const elapsed = (performance.now() - start).toFixed(2);
    console.log(`  ✅ Source: ${result.source}`);
    if (result.similarity) {
      console.log(`  📐 Similarity: ${result.similarity.toFixed(4)}`);
    }
    console.log(`  ⏱️  Time: ${elapsed}ms`);
    console.log(`  📦 Results: ${result.items.length} items`);
    console.log();
  }

  // Show cache stats
  const stats = await cache.getStats();
  console.log('📈 Cache Statistics:');
  console.log(`  Total queries cached: ${stats.totalQueries}`);
  console.log(`  Total cache hits: ${stats.totalHits}`);
  console.log(`  Hit rate: ${(stats.hitRate * 100).toFixed(1)}%`);

  const dbStats = await getCacheStats(db);
  console.log(`\n💾 Database Statistics:`);
  console.log(`  Query documents: ${dbStats.queryCount}`);
  console.log(`  Result documents: ${dbStats.resultCount}`);
}

// Run demo if executed directly
if (import.meta.main) {
  demo().catch(console.error);
}
