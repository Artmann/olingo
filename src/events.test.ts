/* eslint-disable @typescript-eslint/require-await */
import { describe, it, expect, afterEach, vi } from 'vitest'
import { existsSync } from 'node:fs'
import { unlink, rm } from 'node:fs/promises'
import { EmbeddingEngine } from './engine'
import type { EmbeddingProvider } from './types'

const testStorePath = './test-data/events-test.raptor'

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

describe('Events / Hooks', () => {
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

  it('should emit store event with key', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    const listener = vi.fn()
    engine.on('store', listener)

    try {
      await engine.store('doc1', 'hello')
      expect(listener).toHaveBeenCalledOnce()
      expect(listener).toHaveBeenCalledWith({ key: 'doc1' })
    } finally {
      await engine.dispose()
    }
  })

  it('should emit delete event with key', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    const listener = vi.fn()
    engine.on('delete', listener)

    try {
      await engine.store('doc1', 'hello')
      await engine.delete('doc1')
      expect(listener).toHaveBeenCalledOnce()
      expect(listener).toHaveBeenCalledWith({ key: 'doc1' })
    } finally {
      await engine.dispose()
    }
  })

  it('should emit search event with query and result count', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    const listener = vi.fn()
    engine.on('search', listener)

    try {
      await engine.store('doc1', 'hello')
      const results = await engine.search('hello', 10, 0)
      expect(listener).toHaveBeenCalledOnce()
      expect(listener).toHaveBeenCalledWith({
        query: 'hello',
        resultCount: results.length
      })
    } finally {
      await engine.dispose()
    }
  })

  it('should emit update event with key', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    const listener = vi.fn()
    engine.on('update', listener)

    try {
      await engine.store('doc1', 'hello')
      await engine.update('doc1', 'world')
      expect(listener).toHaveBeenCalledOnce()
      expect(listener).toHaveBeenCalledWith({ key: 'doc1' })
    } finally {
      await engine.dispose()
    }
  })

  it('should support multiple listeners', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    const listener1 = vi.fn()
    const listener2 = vi.fn()
    engine.on('store', listener1)
    engine.on('store', listener2)

    try {
      await engine.store('doc1', 'hello')
      expect(listener1).toHaveBeenCalledOnce()
      expect(listener2).toHaveBeenCalledOnce()
    } finally {
      await engine.dispose()
    }
  })

  it('should support removeListener', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    const listener = vi.fn()
    engine.on('store', listener)
    engine.removeListener('store', listener)

    try {
      await engine.store('doc1', 'hello')
      expect(listener).not.toHaveBeenCalled()
    } finally {
      await engine.dispose()
    }
  })
})
