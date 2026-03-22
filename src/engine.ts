import {
  getLlama,
  LlamaLogLevel,
  resolveModelFile,
  type Llama,
  type LlamaModel,
  type LlamaEmbeddingContext
} from 'node-llama-cpp'
import { EventEmitter } from 'node:events'
import invariant from 'tiny-invariant'

import {
  StorageEngine,
  ensureV2Format,
  opType,
  ReadOnlyError,
  verifyDatabase
} from './storage-engine'
import type { CompactionResult, VerifyResult } from './storage-engine'
import { KeyNotFoundError } from './key-not-found-error'
import { HnswIndex } from './storage-engine/hnsw-index'
import {
  serializeHnswIndex,
  deserializeHnswIndex
} from './storage-engine/hnsw-persistence'
import { LRUCache } from './lru-cache'
import {
  copyFile,
  readFile,
  writeFile,
  mkdir as mkdirFs,
  stat
} from 'node:fs/promises'
import { dirname } from 'node:path'
import type {
  DatabaseStats,
  DetailedSearchResult,
  EmbeddingEntry,
  EmbeddingProvider,
  EngineOptions,
  SearchOptions,
  SearchResult
} from './types'

// Model URI for bge-small-en-v1.5 GGUF (384 dimensions, ~67MB)
const defaultModelUri =
  'hf:CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q8_0.gguf'
const defaultCacheDir = './.cache/models'
const defaultDimension = 384

/**
 * Error thrown when the embedding model fails to initialize.
 * This can happen due to missing native bindings, network issues, or invalid model files.
 */
export class ModelInitializationError extends Error {
  constructor(
    message: string,
    public readonly cause?: Error
  ) {
    super(message)
    this.name = 'ModelInitializationError'
  }
}

/**
 * Error thrown when embedding generation fails.
 */
export class EmbeddingGenerationError extends Error {
  constructor(
    message: string,
    public readonly cause?: Error
  ) {
    super(message)
    this.name = 'EmbeddingGenerationError'
  }
}

// Static set of engines with loaded native resources, used by shared exit handler
const enginesWithNativeResources = new Set<EmbeddingEngine>()
let exitHandlerRegistered = false

/**
 * Shared exit handler that disposes native resources for all active engines.
 * This prevents segfaults when process.exit() is called without disposing engines.
 * Uses a single listener to avoid adding multiple exit handlers.
 *
 * Note: We call dispose() without await since exit handlers must be synchronous.
 * The native dispose calls may still complete before process termination.
 */
function handleProcessExit(): void {
  for (const engine of enginesWithNativeResources) {
    engine._disposeNativeResourcesSync()
  }
  enginesWithNativeResources.clear()
}

export class EmbeddingEngine extends EventEmitter {
  private storageEngine: StorageEngine | null = null
  private storePath: string
  private cacheDir: string
  private dimension: number
  private readonly readOnly: boolean
  private llama?: Llama
  private model?: LlamaModel
  private embeddingContext?: LlamaEmbeddingContext
  private initPromise?: Promise<void>
  private storageInitPromise?: Promise<StorageEngine>
  // HNSW index for approximate nearest-neighbour search
  private hnswIndex: HnswIndex | null = null
  // LRU cache for text-to-embedding lookups (avoids regenerating embeddings for repeated text)
  private textEmbeddingCache: LRUCache<string, Float32Array> | null = null
  // Custom embedding provider (when set, the default model is not loaded)
  private readonly customProvider: EmbeddingProvider | null = null

  constructor(options: EngineOptions) {
    super()
    this.storePath = options.storePath
    this.cacheDir = options.cacheDir ?? defaultCacheDir
    this.readOnly = options.readOnly ?? false
    if (options.embeddingProvider) {
      this.customProvider = options.embeddingProvider
      this.dimension = options.embeddingProvider.dimension
    } else {
      this.dimension = defaultDimension
    }
    if (options.embeddingCacheSize && options.embeddingCacheSize > 0) {
      this.textEmbeddingCache = new LRUCache(options.embeddingCacheSize)
    }
  }

  /**
   * Gets or initializes the storage engine
   * Performs migration from v1 format if needed
   */
  private async ensureStorageEngine(): Promise<StorageEngine> {
    if (this.storageEngine) {
      return this.storageEngine
    }

    // Prevent concurrent initialization
    if (this.storageInitPromise) {
      return this.storageInitPromise
    }

    this.storageInitPromise = this.initializeStorage()
    this.storageEngine = await this.storageInitPromise
    return this.storageEngine
  }

  private async initializeStorage(): Promise<StorageEngine> {
    // Check and migrate from v1 format if needed (skip for read-only mode)
    if (!this.readOnly) {
      await ensureV2Format(this.storePath, this.dimension)
    }

    // Create storage engine
    return StorageEngine.create({
      dataPath: this.storePath,
      dimension: this.dimension,
      readOnly: this.readOnly
    })
  }

  /**
   * Lazily builds HNSW index from storage on first call
   */
  private get hnswIndexPath(): string {
    return this.storePath + '-hnsw'
  }

  private async ensureHnswIndex(storage: StorageEngine): Promise<HnswIndex> {
    if (this.hnswIndex !== null) {
      return this.hnswIndex
    }

    // Try loading from disk first
    try {
      const data = await readFile(this.hnswIndexPath)
      const index = deserializeHnswIndex(new Uint8Array(data))

      // Verify the index has the same number of entries as the storage
      // If not, rebuild from scratch (stale sidecar)
      if (index.has('__check__') || this.isHnswStale(index, storage)) {
        throw new Error('stale')
      }

      this.hnswIndex = index
      return this.hnswIndex
    } catch {
      // Sidecar missing or stale, rebuild from storage
    }

    const index = new HnswIndex()

    for (const [key, location] of storage.locations()) {
      const embedding = await storage.readEmbeddingAt(location.offset)
      if (embedding) {
        index.insert(key, Array.from(embedding))
      }
    }

    this.hnswIndex = index

    // Persist to disk (non-blocking, don't fail if it can't write)
    this.persistHnswIndex(index).catch(() => {})

    return this.hnswIndex
  }

  private isHnswStale(index: HnswIndex, storage: StorageEngine): boolean {
    // Simple check: does the index key count match storage?
    const storageKeys = new Set(storage.keys())
    for (const key of storageKeys) {
      if (!index.has(key)) {
        return true
      }
    }
    return false
  }

  private async persistHnswIndex(index: HnswIndex): Promise<void> {
    if (this.readOnly) {
      return
    }
    const data = serializeHnswIndex(index)
    await writeFile(this.hnswIndexPath, data)
  }

  /**
   * Gets or initializes the embedding model
   * Caches the model instance to avoid repeated initialization overhead
   */
  private async ensureModelLoaded(): Promise<void> {
    if (this.embeddingContext) {
      return
    }

    // Prevent concurrent initialization
    if (this.initPromise) {
      return this.initPromise
    }

    this.initPromise = this.initializeModel()
    await this.initPromise
  }

  private async initializeModel(): Promise<void> {
    // Step 1: Initialize llama runtime
    try {
      this.llama = await getLlama({
        logLevel: LlamaLogLevel.error // Suppress tokenizer warnings for embedding models
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      throw new ModelInitializationError(
        `Failed to initialize llama runtime. This usually means the native bindings ` +
          `are not available for your platform.\n\n` +
          `Original error: ${message}\n\n` +
          `Troubleshooting:\n` +
          `  - Ensure you have a supported platform (Windows x64, macOS arm64/x64, Linux x64)\n` +
          `  - Try reinstalling: npm rebuild node-llama-cpp\n` +
          `  - Check that your Node.js version is supported (18+)`,
        error instanceof Error ? error : undefined
      )
    }

    // Step 2: Download/resolve the model file
    let modelPath: string
    try {
      modelPath = await resolveModelFile(defaultModelUri, this.cacheDir)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      throw new ModelInitializationError(
        `Failed to download or locate the embedding model.\n\n` +
          `Model: ${defaultModelUri}\n` +
          `Cache directory: ${this.cacheDir}\n` +
          `Original error: ${message}\n\n` +
          `Troubleshooting:\n` +
          `  - Check your internet connection\n` +
          `  - Ensure the cache directory is writable: ${this.cacheDir}\n` +
          `  - Try deleting the cache directory and retrying`,
        error instanceof Error ? error : undefined
      )
    }

    // Step 3: Load the model
    try {
      this.model = await this.llama.loadModel({
        modelPath
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      throw new ModelInitializationError(
        `Failed to load the embedding model.\n\n` +
          `Model path: ${modelPath}\n` +
          `Original error: ${message}\n\n` +
          `Troubleshooting:\n` +
          `  - The model file may be corrupted, try deleting: ${modelPath}\n` +
          `  - Ensure you have enough memory (~200MB required)\n` +
          `  - Check that the model file is a valid GGUF format`,
        error instanceof Error ? error : undefined
      )
    }

    // Step 4: Create embedding context
    try {
      this.embeddingContext = await this.model.createEmbeddingContext()
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      throw new ModelInitializationError(
        `Failed to create embedding context.\n\n` +
          `Original error: ${message}\n\n` +
          `This model may not support embeddings or there may be insufficient memory.`,
        error instanceof Error ? error : undefined
      )
    }

    // Register this engine for cleanup on process exit
    enginesWithNativeResources.add(this)
    if (!exitHandlerRegistered) {
      process.on('exit', handleProcessExit)
      exitHandlerRegistered = true
    }
  }

  /**
   * Truncates text to fit within the model's context size
   * Uses the model's tokenizer for accurate token counting
   * BGE-small supports 512 tokens, we use 500 to leave room for special tokens
   */
  private truncateToContextSize(text: string): string {
    if (!this.model) {
      // Fallback if model not loaded yet
      const maxChars = 300 * 3
      return text.length <= maxChars ? text : text.slice(0, maxChars)
    }

    const maxTokens = 500
    const tokens = this.model.tokenize(text)

    if (tokens.length <= maxTokens) {
      return text
    }

    // Truncate tokens and detokenize
    const truncatedTokens = tokens.slice(0, maxTokens)
    return this.model.detokenize(truncatedTokens)
  }

  /**
   * Generates embedding from text using node-llama-cpp with bge-small-en-v1.5 model
   * @param text - Text to embed
   * @returns 384-dimensional embedding vector (normalized)
   */
  async generateEmbedding(text: string): Promise<number[]> {
    const embedding = await this.generateEmbeddingFloat32(text)
    return Array.from(embedding)
  }

  /**
   * Internal method that returns embedding as Float32Array for performance
   * Uses Float32Array throughout internal operations to avoid boxing overhead
   * Checks the text embedding cache first to avoid regenerating embeddings
   */
  private async generateEmbeddingFloat32(text: string): Promise<Float32Array> {
    // Check cache first
    if (this.textEmbeddingCache) {
      const cached = this.textEmbeddingCache.get(text)
      if (cached) {
        return cached
      }
    }

    let result: Float32Array

    if (this.customProvider) {
      result = await this.customProvider.generateEmbedding(text)
    } else {
      await this.ensureModelLoaded()
      invariant(this.embeddingContext, 'Embedding context not initialized')

      const truncatedText = this.truncateToContextSize(text)

      let embedding
      try {
        embedding = await this.embeddingContext.getEmbeddingFor(truncatedText)
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        const textPreview =
          truncatedText.length > 100
            ? truncatedText.slice(0, 100) + '...'
            : truncatedText
        throw new EmbeddingGenerationError(
          `Failed to generate embedding for text.\n\n` +
            `Text preview: "${textPreview}"\n` +
            `Original error: ${message}\n\n` +
            `This may be caused by invalid input or a model error.`,
          error instanceof Error ? error : undefined
        )
      }

      result = new Float32Array(embedding.vector)
    }

    // Store in cache
    if (this.textEmbeddingCache) {
      this.textEmbeddingCache.set(text, result)
    }

    return result
  }

  /**
   * Retrieves an embedding entry by key
   * O(1) lookup via in-memory index
   * @param key - Unique identifier for the entry
   * @returns The embedding entry, or null if not found
   */
  async get(key: string): Promise<EmbeddingEntry | null> {
    invariant(key, 'Key must be provided.')

    const storage = await this.ensureStorageEngine()
    const record = await storage.readRecord(key)

    if (!record) {
      return null
    }

    return {
      key: record.key,
      text: '', // Text is not stored in v2 format
      embedding: Array.from(record.embedding),
      timestamp: Number(record.timestamp)
    }
  }

  /**
   * Checks if a key exists in the database
   * O(1) lookup via in-memory index
   * @param key - Unique identifier for the entry
   * @returns true if the key exists, false otherwise
   */
  async has(key: string): Promise<boolean> {
    invariant(key, 'Key must be provided.')

    const storage = await this.ensureStorageEngine()
    return storage.hasKey(key)
  }

  /**
   * Searches for similar embeddings using cosine similarity
   * @param query - Text query to search for
   * @param limit - Maximum number of results to return (default: 10)
   * @param minSimilarity - Minimum similarity threshold (default: 0.5, range: 0 to 1)
   * @returns Array of search results sorted by similarity (highest first)
   */
  async search(
    query: string,
    options: SearchOptions & { includeDetails: true }
  ): Promise<DetailedSearchResult[]>
  async search(query: string, options?: SearchOptions): Promise<SearchResult[]>
  async search(
    query: string,
    limit?: number,
    minSimilarity?: number
  ): Promise<SearchResult[]>
  async search(
    query: string,
    limitOrOptions?: number | SearchOptions,
    minSimilarityArg?: number
  ): Promise<SearchResult[] | DetailedSearchResult[]> {
    let limit: number
    let minSimilarity: number
    let includeDetails: boolean

    if (typeof limitOrOptions === 'object' && limitOrOptions !== null) {
      limit = limitOrOptions.limit ?? 10
      minSimilarity = limitOrOptions.minSimilarity ?? 0.5
      includeDetails = limitOrOptions.includeDetails ?? false
    } else {
      limit = limitOrOptions ?? 10
      minSimilarity = minSimilarityArg ?? 0.5
      includeDetails = false
    }

    invariant(query, 'Query text must be provided.')
    invariant(limit > 0, 'Limit must be a positive integer.')
    invariant(
      minSimilarity >= 0 && minSimilarity <= 1,
      'minSimilarity must be between 0 and 1.'
    )

    const storage = await this.ensureStorageEngine()

    if (storage.count() === 0) {
      return []
    }

    const queryEmbedding = await this.generateEmbeddingFloat32(query)
    const hnswIndex = await this.ensureHnswIndex(storage)

    // Overscan to handle minSimilarity filtering post-retrieval
    const overscan = Math.max(limit * 3, limit + 20)
    const candidateKeys = hnswIndex.search(Array.from(queryEmbedding), overscan)

    let queryNorm = 0
    if (includeDetails) {
      for (let i = 0; i < queryEmbedding.length; i++) {
        queryNorm += queryEmbedding[i] * queryEmbedding[i]
      }
      queryNorm = Math.sqrt(queryNorm)
    }

    const results: Array<SearchResult | DetailedSearchResult> = []
    for (const key of candidateKeys) {
      const vec = hnswIndex.getVector(key)
      if (!vec) {
        continue
      }
      const resultEmbedding = new Float32Array(vec)
      const similarity = this.cosineSimilarity(queryEmbedding, resultEmbedding)
      if (similarity >= minSimilarity) {
        if (includeDetails) {
          let dotProduct = 0
          let resultNorm = 0
          for (let i = 0; i < queryEmbedding.length; i++) {
            dotProduct += queryEmbedding[i] * resultEmbedding[i]
            resultNorm += resultEmbedding[i] * resultEmbedding[i]
          }
          resultNorm = Math.sqrt(resultNorm)
          results.push({
            key,
            similarity,
            queryNorm,
            resultNorm,
            dotProduct
          } as DetailedSearchResult)
        } else {
          results.push({ key, similarity })
        }
      }
    }

    results.sort((a, b) => b.similarity - a.similarity)
    const finalResults = results.slice(0, limit)

    this.emit('search', { query, resultCount: finalResults.length })

    return finalResults
  }

  /**
   * Stores a text embedding with WAL-based durability
   * @param key - Unique identifier for this entry
   * @param text - Text to embed and store
   */
  async store(key: string, text: string): Promise<void> {
    if (this.readOnly) {
      throw new ReadOnlyError()
    }
    invariant(key, 'Key must be provided.')
    invariant(text, 'Text must be provided.')

    const embedding = await this.generateEmbeddingFloat32(text)
    const storage = await this.ensureStorageEngine()

    // Determine if this is an insert or update
    const op = storage.hasKey(key) ? opType.update : opType.insert
    await storage.writeRecord(key, embedding, op)

    // Update HNSW index if it exists
    if (this.hnswIndex !== null) {
      this.hnswIndex.insert(key, Array.from(embedding))
    }

    this.emit('store', { key })
  }

  /**
   * Stores multiple text embeddings in batch
   * More efficient than calling store() multiple times
   * Generates embeddings in parallel and writes records sequentially
   * Uses text embedding cache to avoid regenerating embeddings for duplicate texts
   * @param items - Array of {key, text} objects to store
   */
  async storeMany(items: Array<{ key: string; text: string }>): Promise<void> {
    if (this.readOnly) {
      throw new ReadOnlyError()
    }
    invariant(items.length > 0, 'Items array must not be empty.')

    // Generate embeddings in parallel using the unified method
    const embeddingPromises = items.map((item) =>
      this.generateEmbeddingFloat32(item.text)
    )

    const embeddingsList = await Promise.all(embeddingPromises)

    const storage = await this.ensureStorageEngine()

    // Determine operation types before concurrent writes
    const operations = items.map((item) => ({
      key: item.key,
      op: storage.hasKey(item.key) ? opType.update : opType.insert
    }))

    // Write records concurrently (batching will group fsyncs)
    const writePromises = items.map((item, i) => {
      const embedding = embeddingsList[i]
      return storage.writeRecord(item.key, embedding, operations[i].op)
    })

    await Promise.all(writePromises)

    // Update HNSW index after all writes complete
    if (this.hnswIndex !== null) {
      for (let i = 0; i < items.length; i++) {
        this.hnswIndex.insert(items[i].key, Array.from(embeddingsList[i]))
      }
    }
  }

  /**
   * Updates the text for an existing key. Throws KeyNotFoundError if the key doesn't exist.
   * @param key - Unique identifier for the entry to update
   * @param text - New text content to store
   */
  async update(key: string, text: string): Promise<void> {
    if (this.readOnly) {
      throw new ReadOnlyError()
    }
    invariant(key, 'Key must be provided.')
    invariant(text, 'Text must be provided.')

    const storage = await this.ensureStorageEngine()
    if (!storage.hasKey(key)) {
      throw new KeyNotFoundError(key)
    }

    const embedding = await this.generateEmbeddingFloat32(text)
    await storage.writeRecord(key, embedding, opType.update)

    if (this.hnswIndex !== null) {
      this.hnswIndex.insert(key, Array.from(embedding))
    }

    this.emit('update', { key })
  }

  /**
   * Stores a pre-computed embedding directly, bypassing the embedding model.
   * Useful when you have pre-computed embeddings from an external source.
   * @param key - Unique identifier for the entry
   * @param embedding - Pre-computed embedding vector (Float32Array or number[])
   */
  async storeEmbedding(
    key: string,
    embedding: Float32Array | number[]
  ): Promise<void> {
    if (this.readOnly) {
      throw new ReadOnlyError()
    }
    invariant(key, 'Key must be provided.')

    const float32Embedding =
      embedding instanceof Float32Array
        ? embedding
        : new Float32Array(embedding)

    const storage = await this.ensureStorageEngine()
    const op = storage.hasKey(key) ? opType.update : opType.insert
    await storage.writeRecord(key, float32Embedding, op)

    if (this.hnswIndex !== null) {
      this.hnswIndex.insert(key, Array.from(float32Embedding))
    }
  }

  /**
   * Stores multiple pre-computed embeddings in batch.
   * @param items - Array of {key, embedding} objects to store
   */
  async storeManyEmbeddings(
    items: Array<{ key: string; embedding: Float32Array | number[] }>
  ): Promise<void> {
    if (this.readOnly) {
      throw new ReadOnlyError()
    }
    invariant(items.length > 0, 'Items array must not be empty.')

    const storage = await this.ensureStorageEngine()

    const writePromises = items.map((item) => {
      const float32Embedding =
        item.embedding instanceof Float32Array
          ? item.embedding
          : new Float32Array(item.embedding)
      const op = storage.hasKey(item.key) ? opType.update : opType.insert
      return storage.writeRecord(item.key, float32Embedding, op)
    })

    await Promise.all(writePromises)

    if (this.hnswIndex !== null) {
      for (const item of items) {
        const float32Embedding =
          item.embedding instanceof Float32Array
            ? item.embedding
            : new Float32Array(item.embedding)
        this.hnswIndex.insert(item.key, Array.from(float32Embedding))
      }
    }
  }

  /**
   * Deletes an entry by key
   * Logical delete - records a delete marker in the WAL
   * @param key - Unique identifier for the entry to delete
   * @returns true if the entry was deleted, false if it didn't exist
   */
  async delete(key: string): Promise<boolean> {
    if (this.readOnly) {
      throw new ReadOnlyError()
    }
    invariant(key, 'Key must be provided.')

    const storage = await this.ensureStorageEngine()
    const deleted = await storage.deleteRecord(key)

    // Remove from HNSW index if it exists
    if (deleted && this.hnswIndex !== null) {
      this.hnswIndex.delete(key)
    }

    if (deleted) {
      this.emit('delete', { key })
    }

    return deleted
  }

  /**
   * Gets all keys in the database
   * @returns Iterator of all keys
   */
  async keys(): Promise<string[]> {
    const storage = await this.ensureStorageEngine()
    return Array.from(storage.keys())
  }

  /**
   * Returns an async iterator over all keys in the database.
   * More memory-efficient than keys() for large databases.
   */
  /**
   * Searches for multiple queries in batch.
   * Generates embeddings in parallel and runs HNSW lookups for each.
   * @param queries - Array of search query strings
   * @param limit - Maximum results per query (default: 10)
   * @param minSimilarity - Minimum similarity threshold (default: 0.5)
   * @returns Map from query string to its search results
   */
  async searchMany(
    queries: string[],
    limit: number = 10,
    minSimilarity: number = 0.5
  ): Promise<Map<string, SearchResult[]>> {
    const resultMap = new Map<string, SearchResult[]>()

    if (queries.length === 0) {
      return resultMap
    }

    // Deduplicate queries to avoid redundant embedding generation
    const uniqueQueries = [...new Set(queries)]
    const embeddingPromises = uniqueQueries.map((query) =>
      this.generateEmbeddingFloat32(query)
    )
    const uniqueEmbeddings = await Promise.all(embeddingPromises)
    const embeddingMap = new Map<string, Float32Array>()
    for (let i = 0; i < uniqueQueries.length; i++) {
      embeddingMap.set(uniqueQueries[i], uniqueEmbeddings[i])
    }
    const embeddings = queries.map(
      (q) => embeddingMap.get(q) ?? new Float32Array(0)
    )

    // Ensure HNSW index and storage are ready
    const storage = await this.ensureStorageEngine()
    if (storage.count() === 0) {
      for (const query of queries) {
        resultMap.set(query, [])
      }
      return resultMap
    }
    const hnswIndex = await this.ensureHnswIndex(storage)

    // Run lookups sequentially (HNSW is not thread-safe)
    for (let i = 0; i < queries.length; i++) {
      const queryEmbedding = embeddings[i]
      const overscan = Math.max(limit * 3, limit + 20)
      const candidateKeys = hnswIndex.search(
        Array.from(queryEmbedding),
        overscan
      )

      const results: SearchResult[] = []
      for (const key of candidateKeys) {
        const vec = hnswIndex.getVector(key)
        if (!vec) {
          continue
        }
        const similarity = this.cosineSimilarity(
          queryEmbedding,
          new Float32Array(vec)
        )
        if (similarity >= minSimilarity) {
          results.push({ key, similarity })
        }
      }

      results.sort((a, b) => b.similarity - a.similarity)
      resultMap.set(queries[i], results.slice(0, limit))
    }

    return resultMap
  }

  async *keysIterator(): AsyncIterableIterator<string> {
    const storage = await this.ensureStorageEngine()
    for (const key of storage.keys()) {
      yield key
    }
  }

  /**
   * Returns an async iterator that yields search results one at a time.
   * Results are sorted by similarity (highest first).
   * @param query - Search query text
   * @param options - Search options (limit, minSimilarity)
   */
  async *searchStream(
    query: string,
    options?: SearchOptions
  ): AsyncIterableIterator<SearchResult> {
    const results = await this.search(query, options)
    for (const result of results) {
      yield result
    }
  }

  /**
   * Gets the number of entries in the database
   * @returns Number of entries
   */
  async count(): Promise<number> {
    const storage = await this.ensureStorageEngine()
    return storage.count()
  }

  /**
   * Returns database statistics including record count, file sizes, and configuration.
   */
  /**
   * Verify the integrity of the database by scanning all records
   * and validating checksums and structure.
   */
  /**
   * Compact the database by rewriting only live records.
   * Removes dead records and reduces file size.
   */
  /**
   * Create a consistent backup of the database at the given destination path.
   * Copies the data file, WAL, and HNSW sidecar (if they exist).
   * @param destPath - Path for the backup data file (e.g., './backup/db.raptor')
   */
  async backup(destPath: string): Promise<void> {
    // Ensure destination directory exists
    await mkdirFs(dirname(destPath), { recursive: true })

    // Ensure storage is initialized
    await this.ensureStorageEngine()

    // Copy data file
    await copyFile(this.storePath, destPath).catch(() => {})

    // Copy WAL
    await copyFile(this.storePath + '-wal', destPath + '-wal').catch(() => {})

    // Copy HNSW sidecar if it exists
    await copyFile(this.hnswIndexPath, destPath + '-hnsw').catch(() => {})
  }

  async compact(): Promise<CompactionResult> {
    if (this.readOnly) {
      throw new ReadOnlyError()
    }
    const storage = await this.ensureStorageEngine()
    const result = await storage.compact()

    // Rebuild HNSW index after compaction
    this.hnswIndex = null

    return result
  }

  async verify(): Promise<VerifyResult> {
    return verifyDatabase(this.storePath)
  }

  async stats(): Promise<DatabaseStats> {
    const storage = await this.ensureStorageEngine()
    const recordCount = storage.count()

    let dataFileSize = 0
    let walFileSize = 0
    try {
      const dataStat = await stat(this.storePath)
      dataFileSize = dataStat.size
    } catch {
      // File doesn't exist yet
    }
    try {
      const walStat = await stat(this.storePath + '-wal')
      walFileSize = walStat.size
    } catch {
      // WAL doesn't exist yet
    }

    return {
      recordCount,
      dataFileSize,
      walFileSize,
      dimension: this.dimension,
      isReadOnly: this.readOnly
    }
  }

  /**
   * Calculates cosine similarity between two Float32Arrays
   * Uses typed arrays throughout to avoid boxing overhead
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Embeddings must have the same dimensions')
    }

    let dotProduct = 0
    let magnitudeA = 0
    let magnitudeB = 0

    for (let i = 0; i < a.length; i++) {
      const ai = a[i]
      const bi = b[i]
      dotProduct += ai * bi
      magnitudeA += ai * ai
      magnitudeB += bi * bi
    }

    magnitudeA = Math.sqrt(magnitudeA)
    magnitudeB = Math.sqrt(magnitudeB)

    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0
    }

    return dotProduct / (magnitudeA * magnitudeB)
  }

  /**
   * Check if the engine is in read-only mode.
   */
  isReadOnly(): boolean {
    return this.readOnly
  }

  /**
   * Gets statistics about the text embedding cache.
   * @returns Cache stats if enabled, null if cache is disabled
   */
  getTextEmbeddingCacheStats(): { size: number; maxSize: number } | null {
    if (!this.textEmbeddingCache) {
      return null
    }
    return {
      size: this.textEmbeddingCache.size(),
      maxSize: this.textEmbeddingCache.getMaxSize()
    }
  }

  /**
   * Internal method called by the shared exit handler to dispose native resources.
   * Calls dispose without await since exit handlers must be synchronous.
   * @internal
   */
  _disposeNativeResourcesSync(): void {
    try {
      // Call dispose methods without await - they may still complete
      // before process termination and prevent the segfault
      if (this.embeddingContext) {
        void this.embeddingContext.dispose()
        this.embeddingContext = undefined
      }
      if (this.model) {
        void this.model.dispose()
        this.model = undefined
      }
      this.llama = undefined
    } catch {
      // Ignore errors during emergency cleanup
    }
  }

  /**
   * Disposes of resources and closes the storage engine
   * Call this when you're done using the engine to free up memory
   */
  async dispose(): Promise<void> {
    // Remove from shared exit handler tracking
    enginesWithNativeResources.delete(this)

    // Persist HNSW index before clearing
    if (this.hnswIndex !== null) {
      await this.persistHnswIndex(this.hnswIndex).catch(() => {})
      this.hnswIndex = null
    }

    // Clear text embedding cache
    if (this.textEmbeddingCache) {
      this.textEmbeddingCache.clear()
      this.textEmbeddingCache = null
    }

    // Close storage engine
    if (this.storageEngine) {
      await this.storageEngine.close()
      this.storageEngine = null
    }
    this.storageInitPromise = undefined

    // Dispose embedding model
    if (this.embeddingContext) {
      await this.embeddingContext.dispose()
      this.embeddingContext = undefined
    }
    if (this.model) {
      await this.model.dispose()
      this.model = undefined
    }
    this.llama = undefined
    this.initPromise = undefined
  }
}
