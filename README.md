# Semantic Cache for Graph-RAG Pipelines

A high-performance semantic caching layer for Graph-RAG (Retrieval-Augmented Generation) pipelines using **ArangoDB** and **OpenAI embeddings**. Enables instant responses for semantically similar queries by caching query embeddings and their graph retrieval results.

## Features

- **Semantic Similarity Matching** - Finds cached results for paraphrased queries using cosine similarity
- **Vector Index Support** - Uses ArangoDB 3.12.4+ native vector indexes (Faiss-based) for O(log n) lookups
- **Graph-RAG Integration** - Caches both vector search results and graph traversal paths
- **TTL-based Expiration** - Automatic cache invalidation with volatility-aware TTL
- **Model Version Tracking** - Cache invalidation when embedding/reranker models change
- **Multi-tenant Support** - Isolated caching per tenant
- **Intent Logging** - Extracts entities, facets, and time references for analytics

## Installation

```bash
bun install
```

## Environment Variables

Create a `.env` file:

```env
# OpenAI API Key (for embeddings)
OPENAI_API_KEY=sk-...

# ArangoDB Configuration
ARANGODB_URL=https://your-arangodb-instance.com
ARANGODB_DATABASE=your_database
ARANGODB_USERNAME=your_username
ARANGODB_PASSWORD=your_password
```

## Quick Start

### Basic Usage

```typescript
import { createDatabase, setupSemanticCache, SemanticCache } from './src';

// Initialize
const db = createDatabase();
await setupSemanticCache(db);

// Create cache instance
const cache = new SemanticCache(db, {
  similarityThreshold: 0.85,  // 85% similarity to match
  embeddingModel: 'text-embedding-3-small',
  defaultTtlMs: 7 * 24 * 60 * 60 * 1000,  // 7 days
});

// Use with your retrieval function
const result = await cache.retrieve(
  'What is the price of iPhone 15?',
  async (normalizedQuery, queryVector) => {
    // Your Graph-RAG retrieval logic here
    return [
      { id: 'products/iphone-15', type: 'node', score: 0.95 },
      { id: 'categories/smartphones', type: 'node', score: 0.82 },
    ];
  }
);

console.log(result.source);  // 'fresh' or 'semantic-cache'
console.log(result.items);   // Retrieved items
```

### With Graph-RAG

```typescript
import {
  createDatabase,
  setupSemanticCache,
  SemanticCache,
  setupKnowledgeGraph,
  seedKnowledgeGraph,
  graphRAGRetrieval
} from './src';

const db = createDatabase();
await setupSemanticCache(db);
await setupKnowledgeGraph(db);
await seedKnowledgeGraph(db);

const cache = new SemanticCache(db);

const result = await cache.retrieve(
  'Samsung Galaxy S24 specs',
  (query, vec) => graphRAGRetrieval(db, query, vec, {
    topK: 10,
    graphDepth: 2,
    includeCategories: true,
    includeFeatures: true,
    includeRelated: true,
  })
);
```

## Running the Demos

```bash
# Simple cache demo
bun run src/index.ts

# Full Graph-RAG demo with knowledge graph
bun run src/demo-graph-rag.ts

# Vector index test
bun run src/test-vector-index.ts

# Run tests
bun test
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Normalize + Embed                          │
│              (OpenAI text-embedding-3-small)                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Semantic Cache Lookup (ArangoDB)                │
│    ┌────────────────────────────────────────────────────┐   │
│    │  COSINE_SIMILARITY(cached_vec, query_vec) >= 0.85  │   │
│    │  + Vector Index (Faiss IVF) for large collections  │   │
│    └────────────────────────────────────────────────────┘   │
└───────────────┬─────────────────────────┬───────────────────┘
                │                         │
          Cache HIT                  Cache MISS
                │                         │
                ▼                         ▼
┌───────────────────────┐   ┌─────────────────────────────────┐
│  Return cached items  │   │      Graph-RAG Retrieval        │
│  (instant response)   │   │  ┌───────────────────────────┐  │
└───────────────────────┘   │  │ 1. Vector similarity on   │  │
                            │  │    product embeddings     │  │
                            │  │ 2. Graph traversal for    │  │
                            │  │    related nodes/edges    │  │
                            │  │ 3. Score aggregation      │  │
                            │  └───────────────────────────┘  │
                            │              │                   │
                            │              ▼                   │
                            │    Store in cache + return      │
                            └─────────────────────────────────┘
```

## ArangoDB Collections

### Cache Collections

| Collection | Type | Purpose |
|------------|------|---------|
| `sc_queries` | Document | Cached query embeddings and metadata |
| `sc_results` | Document | Cached retrieval results with TTL |

### Knowledge Graph Collections (Demo)

| Collection | Type | Purpose |
|------------|------|---------|
| `kg_products` | Vertex | Product nodes with embeddings |
| `kg_categories` | Vertex | Category nodes |
| `kg_features` | Vertex | Feature nodes |
| `kg_belongs_to` | Edge | Product → Category |
| `kg_has_feature` | Edge | Product → Feature |
| `kg_related_to` | Edge | Product ↔ Product |

## Vector Index

ArangoDB 3.12.4+ with `--vector-index` startup option enables native vector indexes:

```typescript
import { createVectorIndex } from './src';

// Create after populating data (Faiss requires training)
await createVectorIndex(db, 1536, {
  metric: 'cosine',
  defaultNProbe: 10,
  trainingIterations: 25,
});
```

### Index Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dimension` | 1536 | Vector dimension (must match embedding model) |
| `metric` | 'cosine' | Similarity metric: cosine, l2, innerProduct |
| `nLists` | N/15 | Number of Voronoi cells (auto-calculated) |
| `defaultNProbe` | 10 | Cells to search (higher = better but slower) |
| `trainingIterations` | 25 | Training iterations for index |

### Performance

| Cache Size | Index Used | Query Time |
|------------|------------|------------|
| < 100 | Scan | ~10ms |
| 1,000 | Vector | ~5ms |
| 10,000 | Vector | ~10ms |
| 100,000 | Vector | ~20ms |
| 1,000,000 | Vector | ~50ms |

## Configuration

```typescript
interface SemanticCacheConfig {
  // Similarity threshold (0.83-0.88 recommended for paraphrases)
  similarityThreshold: number;

  // OpenAI embedding model
  embeddingModel: 'text-embedding-3-small' | 'text-embedding-3-large';

  // Cache TTL in milliseconds
  defaultTtlMs: number;

  // Number of results to cache vs return
  topKCached: number;   // default: 25
  topKReturned: number; // default: 10

  // Multi-tenant isolation
  tenantId?: string;
}
```

> **Note:** Vector search quality (`nProbe`) is configured at the index level via `createVectorIndex()`, not in the cache config.

### TTL Presets

```typescript
const TTL_PRESETS = {
  static: 30 * 24 * 60 * 60 * 1000,     // 30 days (product specs)
  semiDynamic: 7 * 24 * 60 * 60 * 1000, // 7 days (default)
  dynamic: 24 * 60 * 60 * 1000,          // 24 hours (prices, reviews)
  realtime: 2 * 60 * 60 * 1000,          // 2 hours (news, live data)
};
```

## API Reference

### SemanticCache

```typescript
class SemanticCache {
  constructor(db: Database, config?: Partial<SemanticCacheConfig>);

  // Main retrieval with caching
  retrieve(
    query: string,
    retrieveFn: (normalized: string, vec: number[]) => Promise<CacheItem[]>,
    tenantId?: string
  ): Promise<RetrievalResult>;

  // Return cached + refresh in background
  retrieveWithBackgroundRefresh(
    query: string,
    retrieveFn: (normalized: string, vec: number[]) => Promise<CacheItem[]>,
    tenantId?: string
  ): Promise<RetrievalResult>;

  // Cache statistics
  getStats(): Promise<{ totalQueries, totalHits, hitRate }>;

  // Cache management
  clear(): Promise<void>;
  invalidateByModelRev(modelRev: string): Promise<number>;
}
```

### Database Functions

```typescript
// Setup
createDatabase(): Database;
setupSemanticCache(db: Database, vectorDimension?: number): Promise<void>;
dropSemanticCache(db: Database): Promise<void>;

// Vector index
createVectorIndex(db: Database, dimension: number, options?: VectorIndexOptions): Promise<boolean>;
hasVectorIndex(db: Database): Promise<boolean>;

// Maintenance
getCacheStats(db: Database): Promise<CacheStats>;
evictOldEntries(db: Database, maxAgeMs?: number): Promise<number>;
cleanupOrphanedQueries(db: Database): Promise<number>;  // Remove queries without results
```

### Embeddings

```typescript
// Generate embeddings
embed(text: string, model?: EmbeddingModel): Promise<number[]>;
embedBatch(texts: string[], model?: EmbeddingModel): Promise<number[][]>;

// Text processing
normalizeText(text: string): string;
extractIntent(text: string): QueryIntent;
cosineSimilarity(a: number[], b: number[]): number;
```

## Example Output

```
📝 ROUND 1: Initial queries (all should be fresh)

Query: "What iPhone should I buy?"
  🔍 Graph-RAG: "what iphone should i buy"
  📊 Found 20 items (nodes + edges)
  🔄 Source: fresh
  ⏱️  Time: 636ms
  📦 Top nodes:
     - kg_products/iphone-15 (score: 0.731)
     - kg_products/iphone-15-pro (score: 0.610)

📝 ROUND 2: Paraphrased queries (should hit cache)

Query: "Which iPhone is the best to buy?"
  ⚡ Source: semantic-cache
  📐 Similarity: 87.3%
  ⏱️  Time: 373ms  ← 40% faster!

📈 Final Statistics:
  Cache hit rate: 37.5%
```

## Project Structure

```
src/
├── types.ts           # TypeScript interfaces
├── db.ts              # ArangoDB setup, indexes, maintenance
├── embeddings.ts      # OpenAI embeddings, normalization
├── cache.ts           # SemanticCache class
├── graph-rag.ts       # Knowledge graph + retrieval
├── index.ts           # Main exports + simple demo
├── demo-graph-rag.ts  # Full Graph-RAG demo
├── test-vector-index.ts # Vector index verification
└── index.test.ts      # Tests (23 tests)
```

## Testing

```bash
bun test
```

Tests cover:
- Text normalization
- Intent extraction
- Cosine similarity calculations
- Embedding generation
- Cache hit/miss behavior
- Tenant isolation
- Statistics tracking

## License

MIT
