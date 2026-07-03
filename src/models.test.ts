import { describe, it, expect } from 'vitest'
import {
  isModelPresetName,
  modelPresetNames,
  modelPresets,
  resolveModelConfig
} from './models'

describe('resolveModelConfig', () => {
  it('should resolve to bge-small-en when no model is given', () => {
    const config = resolveModelConfig()
    expect(config).toEqual(modelPresets['bge-small-en'])
    expect(config.dimension).toBe(384)
    expect(config.maxTokens).toBe(500)
  })

  it('should resolve the bge-small-en preset', () => {
    const config = resolveModelConfig('bge-small-en')
    expect(config.uri).toBe(
      'hf:CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q8_0.gguf'
    )
    expect(config.dimension).toBe(384)
    expect(config.maxTokens).toBe(500)
  })

  it('should resolve the bge-m3 preset', () => {
    const config = resolveModelConfig('bge-m3')
    expect(config.uri).toBe('hf:gpustack/bge-m3-GGUF/bge-m3-Q8_0.gguf')
    expect(config.dimension).toBe(1024)
    expect(config.maxTokens).toBe(8000)
  })

  it('should throw for an unknown preset name with available presets listed', () => {
    expect(() => resolveModelConfig('bge-huge' as never)).toThrow(
      /Unknown embedding model preset 'bge-huge'.*bge-small-en.*bge-m3/
    )
  })

  it('should pass through a custom model config', () => {
    const config = resolveModelConfig({
      uri: 'hf:example/custom-gguf/custom-q8_0.gguf',
      dimension: 768,
      maxTokens: 2000
    })
    expect(config.uri).toBe('hf:example/custom-gguf/custom-q8_0.gguf')
    expect(config.dimension).toBe(768)
    expect(config.maxTokens).toBe(2000)
  })

  it('should default maxTokens to 500 for custom configs', () => {
    const config = resolveModelConfig({
      uri: 'hf:example/custom-gguf/custom-q8_0.gguf',
      dimension: 768
    })
    expect(config.maxTokens).toBe(500)
  })
})

describe('model preset helpers', () => {
  it('should list all preset names', () => {
    expect(modelPresetNames).toEqual(['bge-small-en', 'bge-m3'])
  })

  it('should identify preset names', () => {
    expect(isModelPresetName('bge-m3')).toBe(true)
    expect(isModelPresetName('bge-small-en')).toBe(true)
    expect(isModelPresetName('gpt-4')).toBe(false)
    expect(isModelPresetName('')).toBe(false)
  })
})
