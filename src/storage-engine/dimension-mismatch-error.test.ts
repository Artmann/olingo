/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { describe, it, expect, afterEach } from 'vitest'
import { StorageEngine } from './storage-engine'
import { DimensionMismatchError } from './dimension-mismatch-error'
import {
  createTestPaths,
  cleanup,
  generateRandomEmbedding,
  type TestPaths
} from './integration/helpers'

describe('DimensionMismatchError', () => {
  const testPathsList: TestPaths[] = []

  afterEach(async () => {
    await cleanup(testPathsList)
    testPathsList.length = 0
  })

  it('should reject embeddings with wrong dimensions', async () => {
    const paths = createTestPaths('dim-mismatch')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const wrongDimEmbedding = generateRandomEmbedding(128)

    await expect(engine.writeRecord('doc1', wrongDimEmbedding)).rejects.toThrow(
      DimensionMismatchError
    )

    await engine.close()
  })

  it('should accept embeddings with correct dimensions', async () => {
    const paths = createTestPaths('dim-correct')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const correctEmbedding = generateRandomEmbedding(384)

    await engine.writeRecord('doc1', correctEmbedding)
    const record = await engine.readRecord('doc1')
    expect(record).not.toBeNull()
    expect(record!.key).toBe('doc1')

    await engine.close()
  })

  it('should report expected and actual dimension in error message', async () => {
    const paths = createTestPaths('dim-message')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const wrongDimEmbedding = generateRandomEmbedding(256)

    try {
      await engine.writeRecord('doc1', wrongDimEmbedding)
      expect.fail('Should have thrown DimensionMismatchError')
    } catch (error) {
      expect(error).toBeInstanceOf(DimensionMismatchError)
      const dimError = error as DimensionMismatchError
      expect(dimError.expectedDimension).toBe(384)
      expect(dimError.actualDimension).toBe(256)
      expect(dimError.message).toContain('384')
      expect(dimError.message).toContain('256')
    }

    await engine.close()
  })

  it('should expose getDimension() on storage engine', async () => {
    const paths = createTestPaths('dim-getter')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    expect(engine.getDimension()).toBe(384)

    await engine.close()
  })
})
