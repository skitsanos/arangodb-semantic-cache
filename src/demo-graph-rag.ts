/**
 * Full Graph-RAG + Semantic Cache Demo
 *
 * Demonstrates:
 * 1. Knowledge graph with products, categories, features, and relationships
 * 2. Hybrid retrieval: vector similarity + graph traversal
 * 3. Semantic caching of Graph-RAG results
 */

import { createDatabase, setupSemanticCache } from './db';
import { SemanticCache } from './cache';
import {
  setupKnowledgeGraph,
  seedKnowledgeGraph,
  graphRAGRetrieval,
  GRAPH_COLLECTIONS,
} from './graph-rag';
import { embed, normalizeText } from './embeddings';

async function main() {
  console.log('🚀 Graph-RAG + Semantic Cache Demo\n');
  console.log('='.repeat(60));

  // Initialize database
  const db = createDatabase();
  console.log('✓ Connected to ArangoDB\n');

  // Setup semantic cache
  console.log('📦 Setting up semantic cache...');
  await setupSemanticCache(db);
  console.log();

  // Setup knowledge graph
  console.log('🔗 Setting up knowledge graph...');
  await setupKnowledgeGraph(db);
  await seedKnowledgeGraph(db);
  console.log();

  // Create semantic cache instance
  const cache = new SemanticCache(db, {
    similarityThreshold: 0.85,
    embeddingModel: 'text-embedding-3-small',
    topKCached: 25,
    topKReturned: 10,
  });

  // Clear previous cache for clean demo
  await cache.clear();
  console.log('✓ Cache cleared for demo\n');

  // Graph-RAG retrieval function
  const graphRAG = async (normalizedQuery: string, queryVec: number[]) => {
    console.log(`  🔍 Graph-RAG: "${normalizedQuery}"`);
    const results = await graphRAGRetrieval(db, normalizedQuery, queryVec, {
      topK: 10,
      graphDepth: 2,
      includeCategories: true,
      includeFeatures: true,
      includeRelated: true,
    });
    console.log(`  📊 Found ${results.length} items (nodes + edges)`);
    return results;
  };

  // Test queries - Round 1: Initial queries (all fresh)
  const queries = [
    'What iPhone should I buy?',
    'iPhone 15 Pro features and price',
    'Samsung Galaxy S24 Ultra specs',
    'wireless charging accessories',
  ];

  // Round 2: Paraphrased queries (should hit cache)
  const paraphrasedQueries = [
    'Which iPhone is the best to buy?',      // Similar to query 1
    'iPhone 15 Pro specs and cost',          // Similar to query 2
    'Galaxy S24 Ultra specifications',       // Similar to query 3
    'wireless charger for phone',            // Similar to query 4
  ];

  console.log('='.repeat(60));
  console.log('📝 ROUND 1: Initial queries (all should be fresh)\n');

  for (const query of queries) {
    await runQuery(query, cache, graphRAG);
  }

  console.log('='.repeat(60));
  console.log('📝 ROUND 2: Paraphrased queries (should hit cache)\n');

  for (const query of paraphrasedQueries) {
    await runQuery(query, cache, graphRAG);
  }

  async function runQuery(
    query: string,
    cache: SemanticCache,
    graphRAG: (q: string, v: number[]) => Promise<any>
  ) {
    console.log(`Query: "${query}"`);
    const start = performance.now();

    const result = await cache.retrieve(query, graphRAG);

    const elapsed = (performance.now() - start).toFixed(0);
    const sourceEmoji = result.source === 'semantic-cache' ? '⚡' : '🔄';

    console.log(`  ${sourceEmoji} Source: ${result.source}`);
    if (result.similarity) {
      console.log(`  📐 Similarity: ${(result.similarity * 100).toFixed(1)}%`);
    }
    console.log(`  ⏱️  Time: ${elapsed}ms`);

    // Show top results
    const nodes = result.items.filter((i) => i.type === 'node').slice(0, 3);
    const edges = result.items.filter((i) => i.type === 'edge').slice(0, 2);

    if (nodes.length > 0) {
      console.log('  📦 Top nodes:');
      for (const node of nodes) {
        console.log(`     - ${node.id} (score: ${node.score.toFixed(3)})`);
      }
    }
    if (edges.length > 0) {
      console.log('  🔗 Related edges:');
      for (const edge of edges) {
        console.log(`     - ${edge.id} (score: ${edge.score.toFixed(3)})`);
      }
    }
    console.log();
  }

  // Show final statistics
  console.log('='.repeat(60));
  console.log('📈 Final Statistics:\n');

  const stats = await cache.getStats();
  console.log(`  Queries cached: ${stats.totalQueries}`);
  console.log(`  Total hits: ${stats.totalHits}`);
  console.log(`  Cache hit rate: ${(stats.hitRate * 100).toFixed(1)}%`);

  // Show what's in the cache
  const queriesCol = db.collection('sc_queries');
  const cachedQueries = await db.query(`
    FOR q IN sc_queries
      RETURN { text: q.q_text_norm, hits: q.hit_count }
  `);
  const cached = await cachedQueries.all();

  console.log('\n  Cached queries:');
  for (const q of cached) {
    console.log(`    - "${q.text}" (${q.hits} hits)`);
  }

  // Show graph statistics
  console.log('\n📊 Knowledge Graph Statistics:');
  for (const [name, colName] of Object.entries(GRAPH_COLLECTIONS)) {
    const col = db.collection(colName);
    const count = await col.count();
    console.log(`  ${name}: ${count.count} documents`);
  }

  console.log('\n✅ Demo complete!');
}

main().catch(console.error);
