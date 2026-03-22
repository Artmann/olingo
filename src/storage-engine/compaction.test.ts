import { describe, it, expect, afterEach } from 'vitest'
import { stat } from 'node:fs/promises'
import { StorageEngine } from './storage-engine'
import {
  createTestPaths,
  cleanup,
  generateRandomEmbedding,
  type TestPaths
} from './integration/helpers'

describe('Compaction', () => {
  const testPathsList: TestPaths[] = []

  afterEach(async () => {
    await cleanup(testPathsList)
    testPathsList.length = 0
  })

  it('should compact database with deleted records', async () => {
    const paths = createTestPaths('compact-deleted')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    await engine.writeRecord('doc1', generateRandomEmbedding(384))
    await engine.writeRecord('doc2', generateRandomEmbedding(384))
    await engine.writeRecord('doc3', generateRandomEmbedding(384))
    await engine.deleteRecord('doc2')

    const result = await engine.compact()
    expect(result.recordsBefore).toBe(4) // 3 inserts + 1 delete
    expect(result.recordsAfter).toBe(2) // doc1 + doc3
    expect(result.bytesAfter).toBeLessThan(result.bytesBefore)

    await engine.close()
  })

  it('should preserve all live records after compaction', async () => {
    const paths = createTestPaths('compact-preserve')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const emb1 = generateRandomEmbedding(384)
    const emb3 = generateRandomEmbedding(384)
    await engine.writeRecord('doc1', emb1)
    await engine.writeRecord('doc2', generateRandomEmbedding(384))
    await engine.writeRecord('doc3', emb3)
    await engine.deleteRecord('doc2')

    await engine.compact()

    // Verify live records are accessible
    const record1 = await engine.readRecord('doc1')
    expect(record1).not.toBeNull()

    const record3 = await engine.readRecord('doc3')
    expect(record3).not.toBeNull()

    // Verify deleted record is gone
    const record2 = await engine.readRecord('doc2')
    expect(record2).toBeNull()

    expect(engine.count()).toBe(2)

    await engine.close()
  })

  it('should reduce file size after compaction', async () => {
    const paths = createTestPaths('compact-size')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    for (let i = 0; i < 10; i++) {
      await engine.writeRecord(`doc${i}`, generateRandomEmbedding(384))
    }
    for (let i = 0; i < 5; i++) {
      await engine.deleteRecord(`doc${i}`)
    }

    const sizeBefore = (await stat(paths.dataPath)).size
    await engine.compact()
    const sizeAfter = (await stat(paths.dataPath)).size

    expect(sizeAfter).toBeLessThan(sizeBefore)

    await engine.close()
  })

  it('should handle empty database', async () => {
    const paths = createTestPaths('compact-empty')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const result = await engine.compact()
    expect(result.recordsBefore).toBe(0)
    expect(result.recordsAfter).toBe(0)

    await engine.close()
  })

  it('should allow writes after compaction', async () => {
    const paths = createTestPaths('compact-write-after')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    await engine.writeRecord('doc1', generateRandomEmbedding(384))
    await engine.deleteRecord('doc1')
    await engine.compact()

    // Should be able to write after compaction
    await engine.writeRecord('doc2', generateRandomEmbedding(384))
    const record = await engine.readRecord('doc2')
    expect(record).not.toBeNull()
    expect(engine.count()).toBe(1)

    await engine.close()
  })
})
