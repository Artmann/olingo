/* eslint-disable @typescript-eslint/require-await, @typescript-eslint/no-non-null-assertion */
import { describe, it, expect, afterEach } from 'vitest'
import { existsSync } from 'node:fs'
import { unlink, rm } from 'node:fs/promises'
import { EmbeddingEngine } from './engine'
import type { EmbeddingProvider } from './types'

const testStorePath = './test-data/batch-search-test.raptor'

function createMockProvider(dimension = 384): EmbeddingProvider {
  return {
    dimension,
    async generateEmbedding(text: string): Promise<Float32Array> {
      const embedding = new Float32Array(dimension)
      for (let i = 0; i < dimension; i++) {
        embedding[i] = Math.sin(text.charCodeAt(i % text.length) + i) * 0.5
      }
      return embedding
    }
  }
}

describe('Batch Search', () => {
  afterEach(async () => {
    for (const suffix of ['', '-wal', '.lock']) {
      try {
        const path = testStorePath + suffix
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

  it('should return results for multiple queries', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello world')
      await engine.store('doc2', 'goodbye world')
      await engine.store('doc3', 'test data')

      const results = await engine.searchMany(
        ['hello world', 'goodbye world'],
        10,
        0
      )
      expect(results).toBeInstanceOf(Map)
      expect(results.size).toBe(2)
      expect(results.has('hello world')).toBe(true)
      expect(results.has('goodbye world')).toBe(true)
      expect(results.get('hello world')!.length).toBeGreaterThan(0)
      expect(results.get('goodbye world')!.length).toBeGreaterThan(0)
    } finally {
      await engine.dispose()
    }
  })

  it('should handle empty queries array', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      const results = await engine.searchMany([])
      expect(results).toBeInstanceOf(Map)
      expect(results.size).toBe(0)
    } finally {
      await engine.dispose()
    }
  })

  it('should respect limit and minSimilarity per query', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello')
      await engine.store('doc2', 'world')
      await engine.store('doc3', 'test')

      const results = await engine.searchMany(['hello', 'world'], 1, 0)
      for (const [, queryResults] of results) {
        expect(queryResults.length).toBeLessThanOrEqual(1)
      }
    } finally {
      await engine.dispose()
    }
  })

  it('should deduplicate embedding generation for identical queries', async () => {
    let callCount = 0
    const provider: EmbeddingProvider = {
      dimension: 384,
      async generateEmbedding(text: string): Promise<Float32Array> {
        callCount++
        const embedding = new Float32Array(384)
        for (let i = 0; i < 384; i++) {
          embedding[i] = Math.sin(text.charCodeAt(i % text.length) + i) * 0.5
        }
        return embedding
      }
    }

    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider,
      embeddingCacheSize: 100
    })

    try {
      await engine.store('doc1', 'hello')
      callCount = 0 // Reset after store

      // Search with duplicate queries using a NEW text not in cache
      await engine.searchMany(
        ['different query', 'different query', 'different query'],
        10,
        0
      )

      // With cache enabled, only 1 embedding generation should happen
      // (first call generates + caches, subsequent calls hit cache)
      expect(callCount).toBe(1)
    } finally {
      await engine.dispose()
    }
  })
})
