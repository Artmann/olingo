export {
  EmbeddingEngine,
  ModelInitializationError,
  EmbeddingGenerationError
} from './engine'
export { LRUCache } from './lru-cache'
export {
  ReadOnlyError,
  DatabaseLockedError,
  LockPermissionError,
  DimensionMismatchError
} from './storage-engine'
export type {
  DatabaseStats,
  EmbeddingEntry,
  EmbeddingProvider,
  SearchResult,
  StoreOptions,
  EngineOptions
} from './types'
