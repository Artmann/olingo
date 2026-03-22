/* eslint-disable @typescript-eslint/no-non-null-assertion, @typescript-eslint/require-await */
import { describe, it, expect, afterEach } from 'vitest'
import { existsSync } from 'node:fs'
import { unlink, rm } from 'node:fs/promises'
import { EmbeddingEngine } from './engine'
import type { EmbeddingProvider } from './types'

const testStorePath = './test-data/byoe-test.raptor'

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

function generateEmbedding(dimension: number, seed: number): Float32Array {
  const embedding = new Float32Array(dimension)
  for (let i = 0; i < dimension; i++) {
    embedding[i] = Math.sin(seed + i) * 0.5
  }
  return embedding
}

describe('Bring Your Own Embeddings', () => {
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

  it('should store a pre-computed embedding', async () => {
    const provider = createMockProvider()
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider
    })

    try {
      const embedding = generateEmbedding(384, 42)
      await engine.storeEmbedding('doc1', embedding)

      const entry = await engine.get('doc1')
      expect(entry).not.toBeNull()
      expect(entry!.key).toBe('doc1')
      expect(entry!.embedding.length).toBe(384)
    } finally {
      await engine.dispose()
    }
  })

  it('should reject embedding with wrong dimensions', async () => {
    const provider = createMockProvider(384)
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider
    })

    try {
      const wrongDimEmbedding = generateEmbedding(128, 42)
      await expect(
        engine.storeEmbedding('doc1', wrongDimEmbedding)
      ).rejects.toThrow()
    } finally {
      await engine.dispose()
    }
  })

  it('should store many pre-computed embeddings', async () => {
    const provider = createMockProvider()
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider
    })

    try {
      const items = [
        { key: 'doc1', embedding: generateEmbedding(384, 1) },
        { key: 'doc2', embedding: generateEmbedding(384, 2) },
        { key: 'doc3', embedding: generateEmbedding(384, 3) }
      ]
      await engine.storeManyEmbeddings(items)

      expect(await engine.count()).toBe(3)
      expect(await engine.has('doc1')).toBe(true)
      expect(await engine.has('doc2')).toBe(true)
      expect(await engine.has('doc3')).toBe(true)
    } finally {
      await engine.dispose()
    }
  })

  it('should make stored embeddings searchable', async () => {
    const provider = createMockProvider()
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider
    })

    try {
      // Store some embeddings
      await engine.storeEmbedding('doc1', generateEmbedding(384, 1))
      await engine.storeEmbedding('doc2', generateEmbedding(384, 2))
      await engine.storeEmbedding('doc3', generateEmbedding(384, 100))

      // Search with an embedding close to doc1
      const results = await engine.search('test', 10, 0)
      expect(results.length).toBeGreaterThan(0)
    } finally {
      await engine.dispose()
    }
  })

  it('should accept number[] as well as Float32Array', async () => {
    const provider = createMockProvider()
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider
    })

    try {
      const embedding = Array.from(generateEmbedding(384, 42))
      await engine.storeEmbedding('doc1', embedding)

      const entry = await engine.get('doc1')
      expect(entry).not.toBeNull()
      expect(entry!.embedding.length).toBe(384)
    } finally {
      await engine.dispose()
    }
  })
})
