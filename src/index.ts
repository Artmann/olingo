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
export { KeyNotFoundError } from './key-not-found-error'
export type {
  DatabaseStats,
  EmbeddingEntry,
  EmbeddingProvider,
  SearchResult,
  StoreOptions,
  EngineOptions
} from './types'
