/* eslint-disable @typescript-eslint/require-await */
import { describe, it, expect, afterEach } from 'vitest'
import { existsSync } from 'node:fs'
import { unlink, rm } from 'node:fs/promises'
import { EmbeddingEngine } from './engine'
import type { EmbeddingProvider } from './types'

const testStorePath = './test-data/stats-test.raptor'

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

describe('Database Stats', () => {
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

  it('should return correct stats for empty database', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      const s = await engine.stats()
      expect(s.recordCount).toBe(0)
      expect(s.dimension).toBe(384)
      expect(s.isReadOnly).toBe(false)
      expect(typeof s.dataFileSize).toBe('number')
      expect(typeof s.walFileSize).toBe('number')
    } finally {
      await engine.dispose()
    }
  })

  it('should return correct stats after storing records', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello')
      await engine.store('doc2', 'world')
      await engine.store('doc3', 'test')

      const s = await engine.stats()
      expect(s.recordCount).toBe(3)
      expect(s.dataFileSize).toBeGreaterThan(0)
      expect(s.walFileSize).toBeGreaterThan(0)
    } finally {
      await engine.dispose()
    }
  })

  it('should update stats after delete', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello')
      await engine.store('doc2', 'world')
      await engine.delete('doc1')

      const s = await engine.stats()
      expect(s.recordCount).toBe(1)
    } finally {
      await engine.dispose()
    }
  })

  it('should include file sizes', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello')

      const s = await engine.stats()
      expect(s.dataFileSize).toBeGreaterThan(0)
      expect(s.walFileSize).toBeGreaterThan(0)
    } finally {
      await engine.dispose()
    }
  })
})
