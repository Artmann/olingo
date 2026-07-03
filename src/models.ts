/**
 * Configuration for a GGUF embedding model loaded via node-llama-cpp.
 */
export interface ModelConfig {
  /** Model URI (hf: URIs supported) or local file path */
  uri: string
  /** The dimension of embeddings produced by this model */
  dimension: number
  /** Maximum tokens embedded per text; longer text is truncated (default: 500) */
  maxTokens?: number
}

export interface ResolvedModelConfig extends ModelConfig {
  maxTokens: number
}

const defaultMaxTokens = 500

/**
 * Built-in embedding model presets.
 * bge-small-en is English-only; use bge-m3 for multilingual content.
 */
export const modelPresets = {
  'bge-small-en': {
    uri: 'hf:CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q8_0.gguf',
    dimension: 384,
    // BGE-small supports 512 tokens; leave room for special tokens
    maxTokens: 500
  },
  'bge-m3': {
    uri: 'hf:gpustack/bge-m3-GGUF/bge-m3-Q8_0.gguf',
    dimension: 1024,
    // BGE-M3 supports 8192 tokens; leave room for special tokens
    maxTokens: 8000
  }
} as const satisfies Record<string, ModelConfig>

export type ModelPresetName = keyof typeof modelPresets

/**
 * Model selection: a built-in preset name or a custom GGUF model config.
 */
export type ModelOption = ModelPresetName | ModelConfig

export const modelPresetNames = Object.keys(modelPresets) as ModelPresetName[]

export function isModelPresetName(value: string): value is ModelPresetName {
  return Object.hasOwn(modelPresets, value)
}

export function resolveModelConfig(model?: ModelOption): ResolvedModelConfig {
  if (model === undefined) {
    return modelPresets['bge-small-en']
  }

  if (typeof model === 'string') {
    if (isModelPresetName(model)) {
      return modelPresets[model]
    }
    // The type says ModelPresetName, but the CLI feeds plain strings at runtime
    const name: string = model
    throw new Error(
      `Unknown embedding model preset '${name}'. ` +
        `Available presets: ${modelPresetNames.join(', ')}. ` +
        `For a custom model, pass { uri, dimension } instead.`
    )
  }

  return { ...model, maxTokens: model.maxTokens ?? defaultMaxTokens }
}
