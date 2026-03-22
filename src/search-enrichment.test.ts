/* eslint-disable @typescript-eslint/require-await */
import { describe, it, expect, afterEach } from 'vitest'
import { existsSync } from 'node:fs'
import { unlink, rm } from 'node:fs/promises'
import { EmbeddingEngine } from './engine'
import type { EmbeddingProvider } from './types'

const testStorePath = './test-data/search-enrich-test.raptor'

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

describe('Search Result Enrichment', () => {
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

  it('should return basic results by default', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello world')
      const results = await engine.search('hello world', 10, 0)
      expect(results.length).toBeGreaterThan(0)
      expect(results[0]).toHaveProperty('key')
      expect(results[0]).toHaveProperty('similarity')
      // Should NOT have detailed properties by default
      expect(results[0]).not.toHaveProperty('queryNorm')
    } finally {
      await engine.dispose()
    }
  })

  it('should return detailed results when includeDetails is true', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello world')
      const results = await engine.search('hello world', {
        limit: 10,
        minSimilarity: 0,
        includeDetails: true
      })
      expect(results.length).toBeGreaterThan(0)
      expect(results[0]).toHaveProperty('key')
      expect(results[0]).toHaveProperty('similarity')
      expect(results[0]).toHaveProperty('queryNorm')
      expect(results[0]).toHaveProperty('resultNorm')
      expect(results[0]).toHaveProperty('dotProduct')
    } finally {
      await engine.dispose()
    }
  })

  it('should include correct dotProduct and norms', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello world')
      const results = await engine.search('hello world', {
        limit: 10,
        minSimilarity: 0,
        includeDetails: true
      })
      expect(results.length).toBeGreaterThan(0)
      const r = results[0]
      expect(r.queryNorm).toBeGreaterThan(0)
      expect(r.resultNorm).toBeGreaterThan(0)
      expect(typeof r.dotProduct).toBe('number')

      // Verify: similarity = dotProduct / (queryNorm * resultNorm)
      const expectedSimilarity = r.dotProduct / (r.queryNorm * r.resultNorm)
      expect(r.similarity).toBeCloseTo(expectedSimilarity, 5)
    } finally {
      await engine.dispose()
    }
  })

  it('should maintain backward compatibility with positional args', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello world')
      // Old-style positional args should still work
      const results = await engine.search('hello world', 10, 0)
      expect(results.length).toBeGreaterThan(0)
    } finally {
      await engine.dispose()
    }
  })

  it('should accept SearchOptions as second argument', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello world')
      const results = await engine.search('hello world', {
        limit: 5,
        minSimilarity: 0
      })
      expect(results.length).toBeGreaterThan(0)
      // Without includeDetails, should not have extra props
      expect(results[0]).not.toHaveProperty('queryNorm')
    } finally {
      await engine.dispose()
    }
  })
})
