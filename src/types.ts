import type { ModelOption } from './models'

export interface EmbeddingEntry {
  key: string
  text: string
  embedding: number[]
  timestamp: number
}

export interface DatabaseStats {
  /** Number of live records (excludes deleted) */
  recordCount: number
  /** Size of the data file in bytes */
  dataFileSize: number
  /** Size of the WAL file in bytes */
  walFileSize: number
  /** Embedding dimension */
  dimension: number
  /** Whether the database is in read-only mode */
  isReadOnly: boolean
}

export interface SearchResult {
  key: string
  similarity: number
}

export interface DetailedSearchResult extends SearchResult {
  /** L2 norm of the query embedding */
  queryNorm: number
  /** L2 norm of the result embedding */
  resultNorm: number
  /** Dot product between query and result embeddings */
  dotProduct: number
}

export interface SearchOptions {
  /** Maximum number of results to return (default: 10) */
  limit?: number
  /** Minimum cosine similarity threshold (default: 0.5) */
  minSimilarity?: number
  /** When true, returns DetailedSearchResult with norms and dot product */
  includeDetails?: boolean
}

export interface StoreOptions {
  storePath?: string
}

/**
 * Interface for custom embedding providers.
 * Implement this to use a different embedding model or service.
 */
export interface EmbeddingProvider {
  /** The dimension of embeddings produced by this provider */
  readonly dimension: number
  /** Generate an embedding vector for the given text */
  generateEmbedding(text: string): Promise<Float32Array>
}

export interface EngineOptions {
  storePath: string
  /** Directory to cache downloaded models (default: ./.cache/models) */
  cacheDir?: string
  /** Open database in read-only mode (default: false). Allows concurrent reads without exclusive lock. */
  readOnly?: boolean
  /** Size of the LRU cache for text-to-embedding lookups (default: 0 = disabled) */
  embeddingCacheSize?: number
  /**
   * Embedding model: a built-in preset name ('bge-small-en' | 'bge-m3') or a
   * custom { uri, dimension, maxTokens? } GGUF config (default: 'bge-small-en').
   * Mutually exclusive with embeddingProvider.
   */
  model?: ModelOption
  /** Custom embedding provider. When provided, the default model is not loaded. */
  embeddingProvider?: EmbeddingProvider
}

export interface PackageJson {
  name: string
  version: string
  description: string
}
