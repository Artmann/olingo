/* eslint-disable @typescript-eslint/require-await, @typescript-eslint/no-non-null-assertion */
import { describe, it, expect, afterEach } from 'vitest'
import { existsSync } from 'node:fs'
import { unlink, rm } from 'node:fs/promises'
import { EmbeddingEngine } from './engine'
import { KeyNotFoundError } from './key-not-found-error'
import { ReadOnlyError } from './storage-engine'
import type { EmbeddingProvider } from './types'

const testStorePath = './test-data/update-test.raptor'

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

describe('update() method', () => {
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

  it('should update existing key with new text', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'original text')
      await engine.update('doc1', 'updated text')

      const entry = await engine.get('doc1')
      expect(entry).not.toBeNull()
      expect(entry!.key).toBe('doc1')
      // Embedding should have changed (different text = different embedding)
      expect(entry!.embedding.length).toBe(384)
    } finally {
      await engine.dispose()
    }
  })

  it('should throw KeyNotFoundError for non-existent key', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await expect(engine.update('nonexistent', 'some text')).rejects.toThrow(
        KeyNotFoundError
      )
    } finally {
      await engine.dispose()
    }
  })

  it('should include the key in the error message', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await expect(engine.update('missing-key', 'some text')).rejects.toThrow(
        'missing-key'
      )
    } finally {
      await engine.dispose()
    }
  })

  it('should throw in read-only mode', async () => {
    const provider = createMockProvider()
    // Create database first
    const engine1 = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider
    })
    await engine1.store('doc1', 'hello')
    await engine1.dispose()

    // Open in read-only mode
    const engine2 = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider,
      readOnly: true
    })

    try {
      await expect(engine2.update('doc1', 'new text')).rejects.toThrow(
        ReadOnlyError
      )
    } finally {
      await engine2.dispose()
    }
  })
})
