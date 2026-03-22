/* eslint-disable @typescript-eslint/require-await, @typescript-eslint/no-non-null-assertion */
import { describe, it, expect, afterEach } from 'vitest'
import { existsSync } from 'node:fs'
import { unlink, rm } from 'node:fs/promises'
import { EmbeddingEngine } from './engine'
import type { EmbeddingProvider } from './types'

const testStorePath = './test-data/provider-test.raptor'
const testWalPath = './test-data/provider-test.raptor-wal'
const testLockPath = './test-data/provider-test.raptor.lock'

function createMockProvider(dimension = 384): EmbeddingProvider {
  return {
    dimension,
    async generateEmbedding(text: string): Promise<Float32Array> {
      // Simple deterministic embedding based on text hash
      const embedding = new Float32Array(dimension)
      for (let i = 0; i < dimension; i++) {
        embedding[i] = Math.sin(text.charCodeAt(i % text.length) + i) * 0.5
      }
      return embedding
    }
  }
}

describe('EmbeddingProvider', () => {
  afterEach(async () => {
    for (const path of [testStorePath, testWalPath, testLockPath]) {
      try {
        if (existsSync(path)) {
          await unlink(path)
        }
      } catch {
        // ignore
      }
    }
    try {
      await rm('./test-data', { recursive: true, force: true })
    } catch {
      // ignore
    }
  })

  it('should use custom embedding provider for store', async () => {
    const provider = createMockProvider()
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider
    })

    try {
      await engine.store('doc1', 'hello world')
      const entry = await engine.get('doc1')
      expect(entry).not.toBeNull()
      expect(entry!.key).toBe('doc1')
    } finally {
      await engine.dispose()
    }
  })

  it('should use custom embedding provider for search', async () => {
    const provider = createMockProvider()
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider
    })

    try {
      await engine.store('doc1', 'hello world')
      await engine.store('doc2', 'goodbye world')

      const results = await engine.search('hello world', 10, 0)
      expect(results.length).toBeGreaterThan(0)
      // The exact same text should return highest similarity
      expect(results[0].key).toBe('doc1')
    } finally {
      await engine.dispose()
    }
  })

  it('should validate provider dimension matches database', async () => {
    // First create a database with dimension 384
    const provider384 = createMockProvider(384)
    const engine1 = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider384
    })

    try {
      await engine1.store('doc1', 'hello world')
    } finally {
      await engine1.dispose()
    }

    // Now try to open with dimension 128 provider
    const provider128 = createMockProvider(128)
    const engine2 = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider128
    })

    try {
      await expect(engine2.store('doc2', 'test')).rejects.toThrow()
    } finally {
      await engine2.dispose()
    }
  })

  it('should fall back to default provider when none specified', () => {
    // Just verify construction works without a provider
    const engine = new EmbeddingEngine({
      storePath: testStorePath
    })
    // No error thrown - default provider will be used
    expect(engine).toBeDefined()
  })
})
