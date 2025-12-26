/**
 * Test Vector Index Creation and Search
 *
 * This script:
 * 1. Populates the cache with sample queries
 * 2. Creates the vector index (trained on existing data)
 * 3. Tests similarity search with the index
 * 4. Analyzes the query execution plan
 */

import { createDatabase, setupSemanticCache, createVectorIndex, hasVectorIndex, COLLECTIONS } from './db';
import { SemanticCache } from './cache';
import { embed } from './embeddings';
import { aql } from 'arangojs';
import type { CacheItem } from './types';

async function main() {
  console.log('🧪 Vector Index Test\n');
  console.log('='.repeat(60));

  const db = createDatabase();
  console.log('✓ Connected to ArangoDB\n');

  // Setup collections
  await setupSemanticCache(db);

  // Check if vector index exists
  const hasIndex = await hasVectorIndex(db);
  console.log(`Vector index exists: ${hasIndex}\n`);

  // Create cache and add some sample data
  const cache = new SemanticCache(db, {
    similarityThreshold: 0.85,
  });

  // Clear and repopulate for testing
  await cache.clear();
  console.log('Cache cleared.\n');

  // Sample queries to populate
  const sampleQueries = [
    'What is the price of iPhone 15?',
    'How to reset my password?',
    'Samsung Galaxy S24 specifications',
    'Best laptop for programming',
    'Wireless headphones comparison',
    'How to install Docker on Ubuntu?',
    'Python machine learning tutorial',
    'React vs Vue comparison',
    'PostgreSQL performance optimization',
    'Kubernetes deployment guide',
  ];

  console.log('Populating cache with sample queries...');
  const mockRetriever = async (q: string, v: number[]): Promise<CacheItem[]> => {
    return [{ id: `test/${q.slice(0, 10)}`, type: 'node', score: 0.9 }];
  };

  for (const query of sampleQueries) {
    await cache.retrieve(query, mockRetriever);
    process.stdout.write('.');
  }
  console.log(' Done!\n');

  // Check document count
  const queriesCol = db.collection(COLLECTIONS.queries);
  const count = await queriesCol.count();
  console.log(`Documents in cache: ${count.count}\n`);

  // Create vector index
  console.log('Creating vector index...');
  const indexCreated = await createVectorIndex(db, 1536, {
    metric: 'cosine',
    defaultNProbe: 10,
    trainingIterations: 25,
  });

  if (!indexCreated) {
    console.log('Vector index creation failed or skipped.');
  }

  // List all indexes
  console.log('\nIndexes on sc_queries:');
  const indexes = await queriesCol.indexes();
  for (const idx of indexes) {
    console.log(`  - ${idx.name} (${idx.type})`);
    if (idx.type === 'vector') {
      console.log(`    params: ${JSON.stringify((idx as any).params)}`);
    }
  }

  // Test a similarity search
  console.log('\n' + '='.repeat(60));
  console.log('Testing similarity search:\n');

  const testQuery = 'iPhone 15 cost';
  console.log(`Query: "${testQuery}"`);

  const testVec = await embed(testQuery);
  const start = performance.now();
  const result = await cache.retrieve(testQuery, mockRetriever);
  const elapsed = (performance.now() - start).toFixed(2);

  console.log(`  Source: ${result.source}`);
  if (result.similarity) {
    console.log(`  Similarity: ${(result.similarity * 100).toFixed(1)}%`);
  }
  console.log(`  Time: ${elapsed}ms`);

  // Analyze query execution plan
  console.log('\n' + '='.repeat(60));
  console.log('Query Execution Plan:\n');

  const explainQuery = aql`
    FOR q IN ${queriesCol}
      FILTER q.q_vec != null
      LET sim = COSINE_SIMILARITY(q.q_vec, ${testVec})
      FILTER sim >= 0.85
      SORT sim DESC
      LIMIT 1
      RETURN { key: q._key, sim: sim }
  `;

  try {
    const explanation = await db.explain(explainQuery);
    console.log('Nodes in execution plan:');
    for (const node of explanation.plan.nodes) {
      const indexInfo = (node as any).indexes
        ? ` [indexes: ${(node as any).indexes.map((i: any) => i.type).join(', ')}]`
        : '';
      console.log(`  - ${node.type}${indexInfo}`);
    }

    // Check if vector index is used
    const usesVectorIndex = explanation.plan.nodes.some(
      (node: any) => node.indexes?.some((idx: any) => idx.type === 'vector')
    );
    console.log(`\n✓ Vector index used: ${usesVectorIndex ? 'YES ⚡' : 'NO (brute-force)'}`);
  } catch (err: any) {
    console.log('Could not explain query:', err.message);
  }

  console.log('\n✅ Test complete!');
}

main().catch(console.error);
