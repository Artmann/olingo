/* eslint-disable @typescript-eslint/require-await */
import { describe, it, expect, afterEach } from 'vitest'
import { existsSync } from 'node:fs'
import { unlink, rm } from 'node:fs/promises'
import { EmbeddingEngine } from './engine'
import type { EmbeddingProvider } from './types'

const testStorePath = './test-data/streaming-test.raptor'

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

describe('Streaming/Pagination', () => {
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

  it('should iterate over keys with async iterator', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello')
      await engine.store('doc2', 'world')
      await engine.store('doc3', 'test')

      const keys: string[] = []
      for await (const key of engine.keysIterator()) {
        keys.push(key)
      }
      expect(keys).toHaveLength(3)
      expect(keys).toContain('doc1')
      expect(keys).toContain('doc2')
      expect(keys).toContain('doc3')
    } finally {
      await engine.dispose()
    }
  })

  it('should stream search results one at a time', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello')
      await engine.store('doc2', 'world')

      const results: Array<{ key: string; similarity: number }> = []
      for await (const result of engine.searchStream('hello', {
        minSimilarity: 0
      })) {
        results.push(result)
      }
      expect(results.length).toBeGreaterThan(0)
      expect(results[0]).toHaveProperty('key')
      expect(results[0]).toHaveProperty('similarity')
    } finally {
      await engine.dispose()
    }
  })

  it('should respect limit in streaming search', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello')
      await engine.store('doc2', 'world')
      await engine.store('doc3', 'test')

      const results: Array<{ key: string; similarity: number }> = []
      for await (const result of engine.searchStream('hello', {
        limit: 1,
        minSimilarity: 0
      })) {
        results.push(result)
      }
      expect(results).toHaveLength(1)
    } finally {
      await engine.dispose()
    }
  })

  it('should handle empty database in keysIterator', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      const keys: string[] = []
      for await (const key of engine.keysIterator()) {
        keys.push(key)
      }
      expect(keys).toHaveLength(0)
    } finally {
      await engine.dispose()
    }
  })

  it('should handle empty database in searchStream', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      const results: Array<{ key: string; similarity: number }> = []
      for await (const result of engine.searchStream('hello', {
        minSimilarity: 0
      })) {
        results.push(result)
      }
      expect(results).toHaveLength(0)
    } finally {
      await engine.dispose()
    }
  })
})
