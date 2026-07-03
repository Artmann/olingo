export {
  EmbeddingEngine,
  ModelInitializationError,
  EmbeddingGenerationError
} from './engine'
export { LRUCache } from './lru-cache'
export { modelPresets, modelPresetNames } from './models'
export type { ModelConfig, ModelOption, ModelPresetName } from './models'
export {
  ReadOnlyError,
  DatabaseLockedError,
  LockPermissionError,
  DimensionMismatchError
} from './storage-engine'
export type {
  CompactionResult,
  VerifyResult,
  VerifyIssue
} from './storage-engine'
export { KeyNotFoundError } from './key-not-found-error'
export type {
  DatabaseStats,
  DetailedSearchResult,
  EmbeddingEntry,
  EmbeddingProvider,
  SearchOptions,
  SearchResult,
  StoreOptions,
  EngineOptions
} from './types'
