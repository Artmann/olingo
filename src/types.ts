export interface EmbeddingEntry {
  key: string
  text: string
  embedding: number[]
  timestamp: number
}

export interface SearchResult {
  key: string
  similarity: number
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
  /** Custom embedding provider. When provided, the default model is not loaded. */
  embeddingProvider?: EmbeddingProvider
}

export interface PackageJson {
  name: string
  version: string
  description: string
}
