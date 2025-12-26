# Query Execution Analysis

## Vector Index Status: ✅ ENABLED

ArangoDB 3.12.4+ with `--vector-index` startup option is enabled.

### Vector Index Created

```json
{
  "type": "vector",
  "name": "idx_q_vec_vector",
  "fields": ["q_vec"],
  "params": {
    "dimension": 1536,
    "metric": "cosine",
    "nLists": "N/15 (auto-calculated)",
    "defaultNProbe": 10,
    "trainingIterations": 25
  }
}
```

### How Vector Index Works

1. **Training Phase**: Index is trained on existing data using Faiss IVF (Inverted File)
2. **Clustering**: Vectors are partitioned into `nLists` Voronoi cells
3. **Search**: Only `nProbe` neighboring cells are searched (approximate)

### Query Pattern

```aql
FOR q IN sc_queries
  FILTER q.q_vec != null
  LET sim = COSINE_SIMILARITY(q.q_vec, @queryVec)
  FILTER sim >= 0.85
  SORT sim DESC
  LIMIT 1
  RETURN q
```

### When Vector Index Is Used

| Condition | Index Used? |
|-----------|-------------|
| Collection < 100 docs | ❌ Optimizer may choose scan |
| Collection > 1000 docs | ✅ Vector index used |
| nLists properly tuned | ✅ Best performance |
| SORT by similarity + LIMIT | ✅ Required pattern |

### Performance Characteristics

| Cache Size | Estimated Query Time |
|------------|---------------------|
| 100 | ~5-10ms |
| 1,000 | ~20-50ms |
| 10,000 | ~100-300ms |
| 100,000 | ~1-3 seconds |
| 1,000,000 | ~10-30 seconds ❌ |

---

## Scaling Options

### Option 1: ArangoDB Enterprise Edition

ArangoDB Enterprise includes native vector search capabilities via ArangoSearch:

```js
// Create view with vector analyzer
db._createView("queries_view", "arangosearch", {
  links: {
    sc_queries: {
      fields: {
        q_vec: {
          analyzers: ["identity"],
          features: ["position"]
        }
      }
    }
  }
});

// Query using SEARCH with vector similarity
FOR doc IN queries_view
  SEARCH ANALYZER(
    NEAR(doc.q_vec, @queryVec, @threshold),
    "identity"
  )
  SORT BM25(doc) DESC
  LIMIT 1
  RETURN doc
```

### Option 2: External Vector Store

Use a dedicated vector database alongside ArangoDB:

| Vector Store | Pros | Cons |
|--------------|------|------|
| **Qdrant** | Fast, good filtering | Extra service |
| **Pinecone** | Managed, scalable | Cost, vendor lock-in |
| **Milvus** | Open source, scalable | Complex setup |
| **pgvector** | If already using Postgres | Limited scale |

Architecture:
```
Query → Embed → Vector Store (similarity) → ArangoDB (graph traversal)
```

### Option 3: Hybrid with Redis

Use Redis with RediSearch for vector similarity:

```ts
// Store embeddings in Redis with vector index
await redis.call('FT.CREATE', 'query_idx',
  'ON', 'HASH',
  'SCHEMA', 'q_vec', 'VECTOR', 'HNSW', 6,
  'TYPE', 'FLOAT32', 'DIM', 1536, 'DISTANCE_METRIC', 'COSINE'
);

// Search for similar
const results = await redis.call('FT.SEARCH', 'query_idx',
  '*=>[KNN 1 @q_vec $vec AS score]',
  'PARAMS', 2, 'vec', queryVecBuffer,
  'DIALECT', 2
);
```

### Option 4: In-Memory Cache Layer (Current + Optimization)

For moderate scale, optimize the current approach:

```ts
// Add in-memory LRU cache for hot queries
const hotCache = new Map<string, { vec: number[], results: CacheItem[] }>();

// Pre-filter by normalized text hash before vector search
// Reduces vector comparisons for exact/near-exact matches
```

---

## Recommendations

| Scale | Recommendation |
|-------|----------------|
| **< 10k queries** | Current implementation is fine |
| **10k - 100k** | Add in-memory LRU layer |
| **100k - 1M** | External vector store (Qdrant/Milvus) |
| **1M+** | Dedicated vector DB + clustering |

---

## Verification Commands

Check current index usage:
```bash
curl -s -u "$USER:$PASS" "$URL/_api/explain" \
  -d '{"query": "FOR q IN sc_queries LET sim = COSINE_SIMILARITY(q.q_vec, @v) SORT sim DESC LIMIT 1 RETURN q"}' \
  | jq '.plan.nodes[].type'
```

Check collection size:
```bash
curl -s -u "$USER:$PASS" "$URL/_api/collection/sc_queries/count" | jq '.count'
```

Monitor query performance:
```bash
curl -s -u "$USER:$PASS" "$URL/_api/query/slow" | jq
```
