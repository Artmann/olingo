/* eslint-disable @typescript-eslint/require-await */
import { describe, it, expect, afterEach } from 'vitest'
import { rm } from 'node:fs/promises'
import { EmbeddingEngine } from './engine'
import { DimensionMismatchError } from './storage-engine'
import type { EmbeddingProvider } from './types'

const testStorePath = './test-data/model-option-test.raptor'

// None of these tests store or search real text, so no model is downloaded:
// the engine resolves the dimension synchronously and loads the model lazily.
describe('EngineOptions.model', () => {
  afterEach(async () => {
    await rm('./test-data', { recursive: true, force: true })
  })

  it('should create a 1024-dimension store with the bge-m3 preset', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      model: 'bge-m3'
    })

    try {
      const stats = await engine.stats()
      expect(stats.dimension).toBe(1024)
    } finally {
      await engine.dispose()
    }
  })

  it('should use the dimension from a custom model config', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      model: { uri: 'hf:example/fake-gguf/fake-q8_0.gguf', dimension: 777 }
    })

    try {
      const stats = await engine.stats()
      expect(stats.dimension).toBe(777)
    } finally {
      await engine.dispose()
    }
  })

  it('should default to 384 dimensions when no model is given', async () => {
    const engine = new EmbeddingEngine({ storePath: testStorePath })

    try {
      const stats = await engine.stats()
      expect(stats.dimension).toBe(384)
    } finally {
      await engine.dispose()
    }
  })

  it('should throw when both model and embeddingProvider are given', () => {
    const provider: EmbeddingProvider = {
      dimension: 128,
      async generateEmbedding(): Promise<Float32Array> {
        return new Float32Array(128)
      }
    }

    expect(
      () =>
        new EmbeddingEngine({
          storePath: testStorePath,
          model: 'bge-m3',
          embeddingProvider: provider
        })
    ).toThrow(/Cannot specify both model and embeddingProvider/)
  })

  it('should throw for an unknown preset name', () => {
    expect(
      () =>
        new EmbeddingEngine({
          storePath: testStorePath,
          model: 'word2vec' as never
        })
    ).toThrow(/Unknown embedding model preset 'word2vec'/)
  })

  it('should throw DimensionMismatchError when reopening a store with a different model', async () => {
    // Write one 384-dim record via a mock provider so the file header exists
    // (dimension validation only applies once the database has been written to)
    const provider: EmbeddingProvider = {
      dimension: 384,
      async generateEmbedding(): Promise<Float32Array> {
        return new Float32Array(384).fill(0.5)
      }
    }
    const defaultDimensionEngine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider
    })
    try {
      await defaultDimensionEngine.store('doc1', 'hello')
    } finally {
      await defaultDimensionEngine.dispose()
    }

    const multilingualEngine = new EmbeddingEngine({
      storePath: testStorePath,
      model: 'bge-m3'
    })
    try {
      await expect(multilingualEngine.stats()).rejects.toThrow(
        DimensionMismatchError
      )
      await expect(multilingualEngine.stats()).rejects.toThrow(
        /created with a different embedding model/
      )
    } finally {
      await multilingualEngine.dispose()
    }
  })
})
