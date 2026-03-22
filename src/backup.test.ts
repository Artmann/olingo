/* eslint-disable @typescript-eslint/require-await, @typescript-eslint/no-non-null-assertion */
import { describe, it, expect, afterEach } from 'vitest'
import { existsSync } from 'node:fs'
import { rm } from 'node:fs/promises'
import { EmbeddingEngine } from './engine'
import type { EmbeddingProvider } from './types'

const testStorePath = './test-data/backup-test.raptor'
const backupDir = './test-data/backup-dest'

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

describe('Backup & Restore', () => {
  afterEach(async () => {
    try {
      await rm('./test-data', { recursive: true, force: true })
    } catch {
      // ignore
    }
  })

  it('should create backup with all files', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello')
      await engine.store('doc2', 'world')

      const backupPath = backupDir + '/backup.raptor'
      await engine.backup(backupPath)

      expect(existsSync(backupPath)).toBe(true)
      expect(existsSync(backupPath + '-wal')).toBe(true)
    } finally {
      await engine.dispose()
    }
  })

  it('should create a usable backup', async () => {
    const provider = createMockProvider()
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: provider
    })

    try {
      await engine.store('doc1', 'hello')
      await engine.store('doc2', 'world')

      const backupPath = backupDir + '/usable.raptor'
      await engine.backup(backupPath)

      // Open the backup
      const backupEngine = new EmbeddingEngine({
        storePath: backupPath,
        embeddingProvider: provider
      })

      try {
        const entry = await backupEngine.get('doc1')
        expect(entry).not.toBeNull()
        expect(entry!.key).toBe('doc1')

        expect(await backupEngine.count()).toBe(2)
      } finally {
        await backupEngine.dispose()
      }
    } finally {
      await engine.dispose()
    }
  })

  it('should handle backup to non-existent directory', async () => {
    const engine = new EmbeddingEngine({
      storePath: testStorePath,
      embeddingProvider: createMockProvider()
    })

    try {
      await engine.store('doc1', 'hello')

      const deepPath = backupDir + '/nested/deep/backup.raptor'
      await engine.backup(deepPath)

      expect(existsSync(deepPath)).toBe(true)
    } finally {
      await engine.dispose()
    }
  })
})
