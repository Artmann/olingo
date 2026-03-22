import { describe, it, expect } from 'vitest'
import { HnswIndex } from './hnsw-index'
import { serializeHnswIndex, deserializeHnswIndex } from './hnsw-persistence'

function generateVector(dim: number, seed: number): number[] {
  const vec: number[] = []
  for (let i = 0; i < dim; i++) {
    vec.push(Math.sin(seed + i) * 0.5)
  }
  return vec
}

describe('HNSW Persistence', () => {
  it('should serialize and deserialize HNSW index', () => {
    const index = new HnswIndex()
    index.insert('doc1', generateVector(384, 1))
    index.insert('doc2', generateVector(384, 2))
    index.insert('doc3', generateVector(384, 3))

    const serialized = serializeHnswIndex(index)
    expect(serialized).toBeInstanceOf(Uint8Array)
    expect(serialized.length).toBeGreaterThan(0)

    const deserialized = deserializeHnswIndex(serialized)
    expect(deserialized.has('doc1')).toBe(true)
    expect(deserialized.has('doc2')).toBe(true)
    expect(deserialized.has('doc3')).toBe(true)
  })

  it('should preserve search accuracy after deserialization', () => {
    const index = new HnswIndex()

    // Insert 50 vectors
    for (let i = 0; i < 50; i++) {
      index.insert(`doc${i}`, generateVector(384, i))
    }

    const serialized = serializeHnswIndex(index)
    const deserialized = deserializeHnswIndex(serialized)

    // Search should return same results
    const queryVec = generateVector(384, 0) // Same as doc0
    const originalResults = index.search(queryVec, 5)
    const deserializedResults = deserialized.search(queryVec, 5)

    expect(deserializedResults).toHaveLength(originalResults.length)
    // doc0 should be the top result in both
    expect(originalResults[0]).toBe('doc0')
    expect(deserializedResults[0]).toBe('doc0')
  })

  it('should handle empty index', () => {
    const index = new HnswIndex()

    const serialized = serializeHnswIndex(index)
    const deserialized = deserializeHnswIndex(serialized)

    expect(deserialized.search(generateVector(384, 0), 5)).toHaveLength(0)
  })

  it('should handle single element index', () => {
    const index = new HnswIndex()
    index.insert('doc1', generateVector(384, 1))

    const serialized = serializeHnswIndex(index)
    const deserialized = deserializeHnswIndex(serialized)

    expect(deserialized.has('doc1')).toBe(true)
    const results = deserialized.search(generateVector(384, 1), 5)
    expect(results).toContain('doc1')
  })
})
