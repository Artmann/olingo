/* eslint-disable @typescript-eslint/require-await */
import { describe, it, expect, afterEach } from 'vitest'
import { existsSync } from 'node:fs'
import { unlink, rm } from 'node:fs/promises'
import { EmbeddingEngine } from './engine'
import { KeyNotFoundError } from './key-not-found-error'
import type { EmbeddingProvider } from './types'

const testStorePath = './test-data/similar-to-test.raptor'

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

describe('similarTo', () => {
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

  it('returns other keys ranked by similarity and never the source key', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'machine learning and artificial intelligence')
      await engine.store('doc2', 'machine learning and artificial intelligencx')
      await engine.store('doc3', 'completely unrelated sourdough baking')

      const results = await engine.similarTo('doc1', 10, 0)

      expect(results.length).toBeGreaterThan(0)
      expect(results.every((r) => r.key !== 'doc1')).toBe(true)
      // results are sorted by similarity descending
      for (let i = 1; i < results.length; i++) {
        expect(results[i - 1].similarity).toBeGreaterThanOrEqual(
          results[i].similarity
        )
      }
    } finally {
      await engine.dispose()
    }
  })

  it('respects the limit', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'alpha')
      await engine.store('doc2', 'beta')
      await engine.store('doc3', 'gamma')
      await engine.store('doc4', 'delta')

      const results = await engine.similarTo('doc1', 2, 0)
      expect(results.length).toBeLessThanOrEqual(2)
    } finally {
      await engine.dispose()
    }
  })

  it('filters by minSimilarity', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'alpha')
      await engine.store('doc2', 'beta')
      await engine.store('doc3', 'gamma')

      const results = await engine.similarTo('doc1', { minSimilarity: 0.99 })
      expect(results.every((r) => r.similarity >= 0.99)).toBe(true)
    } finally {
      await engine.dispose()
    }
  })

  it('returns detailed results when includeDetails is true', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello world')
      await engine.store('doc2', 'hello there')

      const results = await engine.similarTo('doc1', {
        minSimilarity: 0,
        includeDetails: true
      })

      expect(results.length).toBeGreaterThan(0)
      expect(results[0]).toHaveProperty('queryNorm')
      expect(results[0]).toHaveProperty('resultNorm')
      expect(results[0]).toHaveProperty('dotProduct')
    } finally {
      await engine.dispose()
    }
  })

  it('throws KeyNotFoundError for an unknown key', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello world')
      await expect(engine.similarTo('nope')).rejects.toThrow(KeyNotFoundError)
    } finally {
      await engine.dispose()
    }
  })

  it('returns an empty array on an empty database', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      const results = await engine.similarTo('anything')
      expect(results).toEqual([])
    } finally {
      await engine.dispose()
    }
  })

  it('supports the options-object overload', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'alpha')
      await engine.store('doc2', 'beta')

      const results = await engine.similarTo('doc1', { limit: 5 })
      expect(Array.isArray(results)).toBe(true)
    } finally {
      await engine.dispose()
    }
  })

  it('emits a similarTo event with key and resultCount', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'alpha')
      await engine.store('doc2', 'beta')

      const events: Array<{ key: string; resultCount: number }> = []
      engine.on(
        'similarTo',
        (payload: { key: string; resultCount: number }) => {
          events.push(payload)
        }
      )

      const results = await engine.similarTo('doc1', 10, 0)

      expect(events.length).toBe(1)
      expect(events[0].key).toBe('doc1')
      expect(events[0].resultCount).toBe(results.length)
    } finally {
      await engine.dispose()
    }
  })
})
