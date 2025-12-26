/**
 * ArangoDB Database Setup and Client
 */

import { Database, aql } from 'arangojs';
import type { SemanticCacheConfig, DEFAULT_CONFIG } from './types';

const {
  ARANGODB_URL,
  ARANGODB_DATABASE,
  ARANGODB_USERNAME,
  ARANGODB_PASSWORD,
} = process.env;

/** Create ArangoDB database connection */
export function createDatabase(): Database {
  const db = new Database({
    url: ARANGODB_URL,
    databaseName: ARANGODB_DATABASE,
    auth: {
      username: ARANGODB_USERNAME!,
      password: ARANGODB_PASSWORD!,
    },
  });
  return db;
}

/** Collection names */
export const COLLECTIONS = {
  queries: 'sc_queries',
  results: 'sc_results',
} as const;

/** Index definitions */
const INDEXES = {
  queries: [
    {
      type: 'persistent' as const,
      fields: ['q_text_norm'],
      name: 'idx_q_text_norm',
      sparse: true,
    },
    {
      type: 'persistent' as const,
      fields: ['last_hit_at'],
      name: 'idx_last_hit_at',
      sparse: true,
    },
    {
      type: 'persistent' as const,
      fields: ['tenant_id', 'q_text_norm'],
      name: 'idx_tenant_norm',
      sparse: true,
    },
  ],
  results: [
    {
      type: 'persistent' as const,
      fields: ['query_id'],
      name: 'idx_query_id',
      unique: true,
    },
    {
      type: 'ttl' as const,
      fields: ['ttl_at'],
      name: 'idx_ttl',
      expireAfter: 0,
    },
    {
      type: 'persistent' as const,
      fields: ['model_rev'],
      name: 'idx_model_rev',
      sparse: true,
    },
  ],
};

/** Setup semantic cache collections and indexes */
export async function setupSemanticCache(
  db: Database,
  vectorDimension: number = 1536
): Promise<void> {
  console.log('Setting up semantic cache collections...');

  // Create collections
  for (const [key, name] of Object.entries(COLLECTIONS)) {
    const collection = db.collection(name);
    const exists = await collection.exists();

    if (!exists) {
      await collection.create();
      console.log(`Created collection: ${name}`);
    } else {
      console.log(`Collection exists: ${name}`);
    }
  }

  // Create indexes for queries collection
  const queriesCollection = db.collection(COLLECTIONS.queries);
  for (const indexDef of INDEXES.queries) {
    try {
      await queriesCollection.ensureIndex(indexDef);
      console.log(`Index ensured: ${COLLECTIONS.queries}.${indexDef.name}`);
    } catch (err: any) {
      if (!err.message?.includes('duplicate')) {
        console.error(`Failed to create index ${indexDef.name}:`, err.message);
      }
    }
  }

  // Vector index will be created after data is populated
  // (Faiss requires training on existing data)
  console.log('Vector index ready (will be created after initial data population)');

  // Create indexes for results collection
  const resultsCollection = db.collection(COLLECTIONS.results);
  for (const indexDef of INDEXES.results) {
    try {
      await resultsCollection.ensureIndex(indexDef as any);
      console.log(`Index ensured: ${COLLECTIONS.results}.${indexDef.name}`);
    } catch (err: any) {
      if (!err.message?.includes('duplicate')) {
        console.error(`Failed to create index ${indexDef.name}:`, err.message);
      }
    }
  }

  console.log('Semantic cache setup complete.');
}

/** Drop semantic cache collections (for testing/reset) */
export async function dropSemanticCache(db: Database): Promise<void> {
  for (const name of Object.values(COLLECTIONS)) {
    const collection = db.collection(name);
    const exists = await collection.exists();
    if (exists) {
      await collection.drop();
      console.log(`Dropped collection: ${name}`);
    }
  }
}

/** Get cache statistics */
export async function getCacheStats(db: Database): Promise<{
  queryCount: number;
  resultCount: number;
  totalHits: number;
  avgHitCount: number;
}> {
  const statsQuery = aql`
    LET queries = (FOR q IN ${db.collection(COLLECTIONS.queries)} COLLECT WITH COUNT INTO c RETURN c)[0]
    LET results = (FOR r IN ${db.collection(COLLECTIONS.results)} COLLECT WITH COUNT INTO c RETURN c)[0]
    LET hits = (FOR q IN ${db.collection(COLLECTIONS.queries)} RETURN q.hit_count)
    RETURN {
      queryCount: queries,
      resultCount: results,
      totalHits: SUM(hits),
      avgHitCount: AVERAGE(hits)
    }
  `;

  const cursor = await db.query(statsQuery);
  const result = await cursor.next();
  return result || { queryCount: 0, resultCount: 0, totalHits: 0, avgHitCount: 0 };
}

/** Create vector index on queries collection (requires existing data) */
export async function createVectorIndex(
  db: Database,
  vectorDimension: number = 1536,
  options: {
    nLists?: number;
    metric?: 'cosine' | 'l2' | 'innerProduct';
    defaultNProbe?: number;
    trainingIterations?: number;
  } = {}
): Promise<boolean> {
  const queriesCollection = db.collection(COLLECTIONS.queries);

  // Check document count - need data for training
  const count = await queriesCollection.count();
  if (count.count === 0) {
    console.log('No documents to train vector index on. Add data first.');
    return false;
  }

  // nLists should be around N/15 per Faiss paper
  const nLists = options.nLists || Math.max(1, Math.floor(count.count / 15));

  try {
    // Check if index already exists
    const indexes = await queriesCollection.indexes();
    const existingVectorIndex = indexes.find(
      (idx: any) => idx.type === 'vector' && idx.fields?.includes('q_vec')
    );

    if (existingVectorIndex) {
      console.log(`Vector index already exists: ${existingVectorIndex.name}`);
      return true;
    }

    // Create vector index
    const indexDef = {
      type: 'vector',
      name: 'idx_q_vec_vector',
      fields: ['q_vec'],
      params: {
        metric: options.metric || 'cosine',
        dimension: vectorDimension,
        nLists: nLists,
        defaultNProbe: options.defaultNProbe || 10,
        trainingIterations: options.trainingIterations || 25,
      },
    };

    console.log(`Creating vector index with nLists=${nLists} (${count.count} docs)...`);
    await queriesCollection.ensureIndex(indexDef as any);
    console.log(`Vector index created: ${COLLECTIONS.queries}.idx_q_vec_vector`);
    return true;
  } catch (err: any) {
    console.error('Failed to create vector index:', err.message);
    return false;
  }
}

/** Check if vector index exists and is usable */
export async function hasVectorIndex(db: Database): Promise<boolean> {
  const queriesCollection = db.collection(COLLECTIONS.queries);
  try {
    const indexes = await queriesCollection.indexes();
    return indexes.some(
      (idx: any) => idx.type === 'vector' && idx.fields?.includes('q_vec')
    );
  } catch {
    return false;
  }
}

/** Evict old cache entries based on last hit time */
export async function evictOldEntries(
  db: Database,
  maxAgeMs: number = 30 * 24 * 60 * 60 * 1000 // 30 days
): Promise<number> {
  const cutoff = Date.now() - maxAgeMs;

  const evictQuery = aql`
    LET oldQueries = (
      FOR q IN ${db.collection(COLLECTIONS.queries)}
        FILTER q.last_hit_at < ${cutoff}
        RETURN q._key
    )

    FOR key IN oldQueries
      LET qid = key
      REMOVE { _key: key } IN ${db.collection(COLLECTIONS.queries)}

    FOR r IN ${db.collection(COLLECTIONS.results)}
      FILTER r.query_id IN oldQueries
      REMOVE r IN ${db.collection(COLLECTIONS.results)}

    RETURN LENGTH(oldQueries)
  `;

  const cursor = await db.query(evictQuery);
  return (await cursor.next()) || 0;
}
