import { isModelPresetName, modelPresetNames } from '../models'
import type { ModelPresetName } from '../models'

export const sharedFlags = {
  storePath: {
    type: String,
    description: 'Path to the embeddings store file',
    default: './database.raptor',
    alias: 's'
  },
  model: {
    type: String,
    description: `Embedding model preset (${modelPresetNames.join(' | ')})`,
    default: 'bge-small-en'
  }
} as const

export const searchFlags = {
  limit: {
    type: Number,
    description: 'Maximum number of results to return',
    default: 10,
    alias: 'l'
  },
  minSimilarity: {
    type: Number,
    description: 'Minimum similarity threshold (0-1)',
    default: 0,
    alias: 'm'
  }
} as const

export function modelFromFlag(value: string): ModelPresetName {
  if (!isModelPresetName(value)) {
    console.error(
      `Unknown model '${value}'. Available models: ${modelPresetNames.join(', ')}`
    )
    process.exit(1)
  }
  return value
}
