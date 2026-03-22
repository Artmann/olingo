/* eslint-disable @typescript-eslint/require-await */
import { describe, it, expect, afterEach } from 'vitest'
import { writeFile, readFile } from 'node:fs/promises'
import { StorageEngine } from './storage-engine'
import { verifyDatabase } from './integrity'
import {
  createTestPaths,
  cleanup,
  generateRandomEmbedding,
  type TestPaths
} from './integration/helpers'

describe('Integrity Check', () => {
  const testPathsList: TestPaths[] = []

  afterEach(async () => {
    await cleanup(testPathsList)
    testPathsList.length = 0
  })

  it('should report healthy for valid database', async () => {
    const paths = createTestPaths('integrity-healthy')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    await engine.writeRecord('doc1', generateRandomEmbedding(384))
    await engine.writeRecord('doc2', generateRandomEmbedding(384))
    await engine.close()

    const result = await verifyDatabase(paths.dataPath)
    expect(result.totalRecords).toBe(2)
    expect(result.validRecords).toBe(2)
    expect(result.corruptRecords).toBe(0)
    expect(result.issues).toHaveLength(0)
  })

  it('should detect corrupted records', async () => {
    const paths = createTestPaths('integrity-corrupt')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    await engine.writeRecord('doc1', generateRandomEmbedding(384))
    await engine.close()

    // Corrupt the data file by flipping some bytes in the middle
    const data = await readFile(paths.dataPath)
    const buf = Buffer.from(data)
    // Corrupt a byte in the middle of the record (after the header)
    const corruptOffset = Math.floor(buf.length / 2)
    buf[corruptOffset] = buf[corruptOffset] ^ 0xff
    await writeFile(paths.dataPath, buf)

    const result = await verifyDatabase(paths.dataPath)
    expect(result.corruptRecords).toBeGreaterThan(0)
    expect(result.issues.length).toBeGreaterThan(0)
  })

  it('should detect truncated records', async () => {
    const paths = createTestPaths('integrity-truncated')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    await engine.writeRecord('doc1', generateRandomEmbedding(384))
    await engine.close()

    // Truncate the data file
    const data = await readFile(paths.dataPath)
    const truncated = data.subarray(0, Math.floor(data.length * 0.7))
    await writeFile(paths.dataPath, truncated)

    const result = await verifyDatabase(paths.dataPath)
    expect(result.corruptRecords).toBeGreaterThan(0)
    expect(result.issues.length).toBeGreaterThan(0)
  })

  it('should report all issues found', async () => {
    const paths = createTestPaths('integrity-issues')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    await engine.writeRecord('doc1', generateRandomEmbedding(384))
    await engine.writeRecord('doc2', generateRandomEmbedding(384))
    await engine.close()

    // Corrupt multiple records
    const data = await readFile(paths.dataPath)
    const buf = Buffer.from(data)
    // Corrupt near the start (first record) and near the end (second record)
    buf[20] = buf[20] ^ 0xff
    buf[buf.length - 20] = buf[buf.length - 20] ^ 0xff
    await writeFile(paths.dataPath, buf)

    const result = await verifyDatabase(paths.dataPath)
    expect(result.issues.length).toBeGreaterThan(0)
    for (const issue of result.issues) {
      expect(issue).toHaveProperty('offset')
      expect(issue).toHaveProperty('message')
    }
  })

  it('should handle non-existent database', async () => {
    const result = await verifyDatabase('./nonexistent.raptor')
    expect(result.totalRecords).toBe(0)
    expect(result.validRecords).toBe(0)
    expect(result.corruptRecords).toBe(0)
  })
})
