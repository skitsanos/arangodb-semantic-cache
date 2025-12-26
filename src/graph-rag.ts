/**
 * Graph-RAG Integration for Semantic Cache
 *
 * This module provides:
 * 1. Sample knowledge graph setup (products, categories, features)
 * 2. Hybrid retrieval: vector similarity + graph traversal
 * 3. Integration with semantic cache
 */

import { Database, aql } from 'arangojs';
import { embed, normalizeText, cosineSimilarity } from './embeddings';
import type { CacheItem } from './types';

/** Graph collection names */
export const GRAPH_COLLECTIONS = {
  // Vertex collections
  products: 'kg_products',
  categories: 'kg_categories',
  features: 'kg_features',
  // Edge collections
  belongsTo: 'kg_belongs_to',      // product -> category
  hasFeature: 'kg_has_feature',    // product -> feature
  relatedTo: 'kg_related_to',      // product <-> product
} as const;

export const GRAPH_NAME = 'knowledge_graph';

/** Product document with embedding */
export interface ProductNode {
  _key: string;
  name: string;
  description: string;
  price: number;
  embedding?: number[];
}

/** Category document */
export interface CategoryNode {
  _key: string;
  name: string;
  description: string;
}

/** Feature document */
export interface FeatureNode {
  _key: string;
  name: string;
  value: string;
}

/** Setup the knowledge graph schema */
export async function setupKnowledgeGraph(db: Database): Promise<void> {
  console.log('Setting up knowledge graph...');

  // Create vertex collections
  for (const name of [GRAPH_COLLECTIONS.products, GRAPH_COLLECTIONS.categories, GRAPH_COLLECTIONS.features]) {
    const col = db.collection(name);
    if (!(await col.exists())) {
      await col.create();
      console.log(`Created vertex collection: ${name}`);
    }
  }

  // Create edge collections
  for (const name of [GRAPH_COLLECTIONS.belongsTo, GRAPH_COLLECTIONS.hasFeature, GRAPH_COLLECTIONS.relatedTo]) {
    const col = db.collection(name);
    if (!(await col.exists())) {
      await col.create({ type: 3 }); // type 3 = edge collection
      console.log(`Created edge collection: ${name}`);
    }
  }

  // Create the named graph
  try {
    const graph = db.graph(GRAPH_NAME);
    if (!(await graph.exists())) {
      await graph.create([
        {
          collection: GRAPH_COLLECTIONS.belongsTo,
          from: [GRAPH_COLLECTIONS.products],
          to: [GRAPH_COLLECTIONS.categories],
        },
        {
          collection: GRAPH_COLLECTIONS.hasFeature,
          from: [GRAPH_COLLECTIONS.products],
          to: [GRAPH_COLLECTIONS.features],
        },
        {
          collection: GRAPH_COLLECTIONS.relatedTo,
          from: [GRAPH_COLLECTIONS.products],
          to: [GRAPH_COLLECTIONS.products],
        },
      ]);
      console.log(`Created graph: ${GRAPH_NAME}`);
    } else {
      console.log(`Graph exists: ${GRAPH_NAME}`);
    }
  } catch (err: any) {
    if (!err.message?.includes('already exists')) {
      console.warn('Graph creation warning:', err.message);
    }
  }

  console.log('Knowledge graph setup complete.');
}

/** Seed sample data into the knowledge graph */
export async function seedKnowledgeGraph(db: Database): Promise<void> {
  console.log('Seeding knowledge graph with sample data...');

  const products = db.collection(GRAPH_COLLECTIONS.products);
  const categories = db.collection(GRAPH_COLLECTIONS.categories);
  const features = db.collection(GRAPH_COLLECTIONS.features);
  const belongsTo = db.collection(GRAPH_COLLECTIONS.belongsTo);
  const hasFeature = db.collection(GRAPH_COLLECTIONS.hasFeature);
  const relatedTo = db.collection(GRAPH_COLLECTIONS.relatedTo);

  // Check if already seeded
  const count = await products.count();
  if (count.count > 0) {
    console.log('Knowledge graph already seeded.');
    return;
  }

  // Categories
  const catSmartphones = await categories.save({
    _key: 'smartphones',
    name: 'Smartphones',
    description: 'Mobile phones with advanced computing capabilities',
  });

  const catAccessories = await categories.save({
    _key: 'accessories',
    name: 'Accessories',
    description: 'Phone accessories and peripherals',
  });

  const catLaptops = await categories.save({
    _key: 'laptops',
    name: 'Laptops',
    description: 'Portable computers',
  });

  // Features
  const features5g = await features.save({ _key: '5g', name: '5G', value: 'Yes' });
  const featuresOled = await features.save({ _key: 'oled', name: 'Display', value: 'OLED' });
  const featuresUsbc = await features.save({ _key: 'usbc', name: 'Charging', value: 'USB-C' });
  const featuresFaceid = await features.save({ _key: 'faceid', name: 'Security', value: 'Face ID' });
  const featuresWireless = await features.save({ _key: 'wireless', name: 'Charging', value: 'Wireless' });

  // Products with embeddings
  const productData = [
    {
      _key: 'iphone-15-pro',
      name: 'iPhone 15 Pro',
      description: 'Apple flagship smartphone with A17 Pro chip, titanium design, and advanced camera system. Features USB-C, 5G connectivity, and ProMotion display.',
      price: 999,
    },
    {
      _key: 'iphone-15',
      name: 'iPhone 15',
      description: 'Apple smartphone with A16 Bionic chip, Dynamic Island, and 48MP camera. USB-C charging and 5G support.',
      price: 799,
    },
    {
      _key: 'samsung-s24-ultra',
      name: 'Samsung Galaxy S24 Ultra',
      description: 'Samsung flagship with Snapdragon 8 Gen 3, S Pen, 200MP camera, and AI features. Titanium frame and 5G.',
      price: 1199,
    },
    {
      _key: 'samsung-s24',
      name: 'Samsung Galaxy S24',
      description: 'Samsung smartphone with Galaxy AI, 50MP camera, and 5G. AMOLED display with 120Hz refresh rate.',
      price: 799,
    },
    {
      _key: 'pixel-8-pro',
      name: 'Google Pixel 8 Pro',
      description: 'Google flagship with Tensor G3 chip, advanced AI photography, 7 years of updates, and pure Android experience.',
      price: 999,
    },
    {
      _key: 'airpods-pro',
      name: 'AirPods Pro 2',
      description: 'Apple wireless earbuds with active noise cancellation, spatial audio, and USB-C charging case.',
      price: 249,
    },
    {
      _key: 'magsafe-charger',
      name: 'MagSafe Charger',
      description: 'Apple magnetic wireless charger for iPhone. Fast wireless charging with perfect alignment.',
      price: 39,
    },
    {
      _key: 'macbook-pro-14',
      name: 'MacBook Pro 14"',
      description: 'Apple laptop with M3 Pro chip, Liquid Retina XDR display, and all-day battery life. For professionals.',
      price: 1999,
    },
  ];

  // Generate embeddings and save products
  console.log('Generating embeddings for products...');
  for (const prod of productData) {
    const embedding = await embed(`${prod.name} ${prod.description}`);
    await products.save({ ...prod, embedding });
  }

  // Create edges: products -> categories
  await belongsTo.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15-pro`, _to: `${GRAPH_COLLECTIONS.categories}/smartphones` });
  await belongsTo.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15`, _to: `${GRAPH_COLLECTIONS.categories}/smartphones` });
  await belongsTo.save({ _from: `${GRAPH_COLLECTIONS.products}/samsung-s24-ultra`, _to: `${GRAPH_COLLECTIONS.categories}/smartphones` });
  await belongsTo.save({ _from: `${GRAPH_COLLECTIONS.products}/samsung-s24`, _to: `${GRAPH_COLLECTIONS.categories}/smartphones` });
  await belongsTo.save({ _from: `${GRAPH_COLLECTIONS.products}/pixel-8-pro`, _to: `${GRAPH_COLLECTIONS.categories}/smartphones` });
  await belongsTo.save({ _from: `${GRAPH_COLLECTIONS.products}/airpods-pro`, _to: `${GRAPH_COLLECTIONS.categories}/accessories` });
  await belongsTo.save({ _from: `${GRAPH_COLLECTIONS.products}/magsafe-charger`, _to: `${GRAPH_COLLECTIONS.categories}/accessories` });
  await belongsTo.save({ _from: `${GRAPH_COLLECTIONS.products}/macbook-pro-14`, _to: `${GRAPH_COLLECTIONS.categories}/laptops` });

  // Create edges: products -> features
  await hasFeature.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15-pro`, _to: `${GRAPH_COLLECTIONS.features}/5g` });
  await hasFeature.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15-pro`, _to: `${GRAPH_COLLECTIONS.features}/oled` });
  await hasFeature.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15-pro`, _to: `${GRAPH_COLLECTIONS.features}/usbc` });
  await hasFeature.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15-pro`, _to: `${GRAPH_COLLECTIONS.features}/faceid` });
  await hasFeature.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15`, _to: `${GRAPH_COLLECTIONS.features}/5g` });
  await hasFeature.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15`, _to: `${GRAPH_COLLECTIONS.features}/usbc` });
  await hasFeature.save({ _from: `${GRAPH_COLLECTIONS.products}/samsung-s24-ultra`, _to: `${GRAPH_COLLECTIONS.features}/5g` });
  await hasFeature.save({ _from: `${GRAPH_COLLECTIONS.products}/samsung-s24-ultra`, _to: `${GRAPH_COLLECTIONS.features}/oled` });
  await hasFeature.save({ _from: `${GRAPH_COLLECTIONS.products}/airpods-pro`, _to: `${GRAPH_COLLECTIONS.features}/usbc` });
  await hasFeature.save({ _from: `${GRAPH_COLLECTIONS.products}/airpods-pro`, _to: `${GRAPH_COLLECTIONS.features}/wireless` });
  await hasFeature.save({ _from: `${GRAPH_COLLECTIONS.products}/magsafe-charger`, _to: `${GRAPH_COLLECTIONS.features}/wireless` });

  // Create edges: related products
  await relatedTo.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15-pro`, _to: `${GRAPH_COLLECTIONS.products}/iphone-15`, relation: 'variant' });
  await relatedTo.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15-pro`, _to: `${GRAPH_COLLECTIONS.products}/airpods-pro`, relation: 'accessory' });
  await relatedTo.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15-pro`, _to: `${GRAPH_COLLECTIONS.products}/magsafe-charger`, relation: 'accessory' });
  await relatedTo.save({ _from: `${GRAPH_COLLECTIONS.products}/samsung-s24-ultra`, _to: `${GRAPH_COLLECTIONS.products}/samsung-s24`, relation: 'variant' });
  await relatedTo.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15-pro`, _to: `${GRAPH_COLLECTIONS.products}/samsung-s24-ultra`, relation: 'competitor' });
  await relatedTo.save({ _from: `${GRAPH_COLLECTIONS.products}/iphone-15-pro`, _to: `${GRAPH_COLLECTIONS.products}/pixel-8-pro`, relation: 'competitor' });
  await relatedTo.save({ _from: `${GRAPH_COLLECTIONS.products}/macbook-pro-14`, _to: `${GRAPH_COLLECTIONS.products}/iphone-15-pro`, relation: 'ecosystem' });

  console.log('Knowledge graph seeded with sample data.');
}

/**
 * Graph-RAG Retrieval Function
 *
 * 1. Vector similarity search on product embeddings
 * 2. Graph traversal to find related nodes (categories, features, related products)
 * 3. Score aggregation and ranking
 */
export async function graphRAGRetrieval(
  db: Database,
  queryText: string,
  queryVec: number[],
  options: {
    topK?: number;
    graphDepth?: number;
    includeCategories?: boolean;
    includeFeatures?: boolean;
    includeRelated?: boolean;
  } = {}
): Promise<CacheItem[]> {
  const {
    topK = 10,
    graphDepth = 2,
    includeCategories = true,
    includeFeatures = true,
    includeRelated = true,
  } = options;

  const products = db.collection(GRAPH_COLLECTIONS.products);

  // Step 1: Vector similarity search on products
  const vectorSearchQuery = aql`
    FOR p IN ${products}
      FILTER p.embedding != null
      LET sim = COSINE_SIMILARITY(p.embedding, ${queryVec})
      FILTER sim > 0.3
      SORT sim DESC
      LIMIT ${topK}
      RETURN {
        id: p._id,
        key: p._key,
        name: p.name,
        score: sim,
        type: 'node'
      }
  `;

  const vectorCursor = await db.query(vectorSearchQuery);
  const vectorResults = await vectorCursor.all();

  if (vectorResults.length === 0) {
    return [];
  }

  const results: CacheItem[] = vectorResults.map((r) => ({
    id: r.id,
    type: 'node' as const,
    score: r.score,
  }));

  // Step 2: Graph traversal from top results
  const topProductIds = vectorResults.slice(0, 3).map((r) => r.id);

  if (includeCategories || includeFeatures || includeRelated) {
    const graphQuery = aql`
      FOR startVertex IN ${topProductIds}
        FOR v, e, p IN 1..${graphDepth} ANY startVertex
          GRAPH ${GRAPH_NAME}
          OPTIONS { bfs: true, uniqueVertices: 'global' }

          LET vertexType = PARSE_IDENTIFIER(v._id).collection
          LET isCategory = vertexType == ${GRAPH_COLLECTIONS.categories}
          LET isFeature = vertexType == ${GRAPH_COLLECTIONS.features}
          LET isProduct = vertexType == ${GRAPH_COLLECTIONS.products}

          FILTER (${includeCategories} AND isCategory)
              OR (${includeFeatures} AND isFeature)
              OR (${includeRelated} AND isProduct AND v._id != startVertex)

          LET pathLength = LENGTH(p.edges)
          LET score = 1.0 / (pathLength + 1)

          RETURN DISTINCT {
            id: v._id,
            type: 'node',
            score: score,
            edgeType: e != null ? PARSE_IDENTIFIER(e._id).collection : null
          }
    `;

    try {
      const graphCursor = await db.query(graphQuery);
      const graphResults = await graphCursor.all();

      // Add graph-traversed nodes with decayed scores
      for (const gr of graphResults) {
        const existing = results.find((r) => r.id === gr.id);
        if (existing) {
          // Boost score if found via multiple paths
          existing.score = Math.min(1, existing.score + gr.score * 0.3);
        } else {
          results.push({
            id: gr.id,
            type: gr.type,
            score: gr.score * 0.5, // Graph-traversed nodes get lower base score
          });
        }
      }

      // Also add edges as results for edge-aware RAG
      const edgeQuery = aql`
        FOR startVertex IN ${topProductIds}
          FOR v, e IN 1..1 ANY startVertex
            GRAPH ${GRAPH_NAME}
            FILTER e != null
            RETURN DISTINCT {
              id: e._id,
              type: 'edge',
              score: 0.4,
              from: e._from,
              to: e._to
            }
      `;

      const edgeCursor = await db.query(edgeQuery);
      const edgeResults = await edgeCursor.all();

      for (const er of edgeResults) {
        results.push({
          id: er.id,
          type: 'edge',
          score: er.score,
        });
      }
    } catch (err) {
      console.warn('Graph traversal failed, returning vector results only:', err);
    }
  }

  // Sort by score and limit
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, topK * 2); // Return more items since we include graph context
}

/** Drop knowledge graph (for testing/reset) */
export async function dropKnowledgeGraph(db: Database): Promise<void> {
  // Drop graph first
  try {
    const graph = db.graph(GRAPH_NAME);
    if (await graph.exists()) {
      await graph.drop(true); // true = also drop collections
      console.log(`Dropped graph: ${GRAPH_NAME}`);
    }
  } catch (err) {
    // Graph might not exist
  }

  // Drop any remaining collections
  for (const name of Object.values(GRAPH_COLLECTIONS)) {
    const col = db.collection(name);
    if (await col.exists()) {
      await col.drop();
      console.log(`Dropped collection: ${name}`);
    }
  }
}
