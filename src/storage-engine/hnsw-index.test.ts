import { describe, it, expect } from 'vitest'
import { HnswIndex } from './hnsw-index'

// ---------------------------------------------------------------------------
// Test subclass that exposes protected internals
// ---------------------------------------------------------------------------

class TestableHnswIndex extends HnswIndex {
  getNodes() {
    return this.nodes
  }
  getEntrypointKey() {
    return this.entrypointKey
  }
}

// ---------------------------------------------------------------------------
// Local brute-force helper for accuracy tests (TC-19, TC-20)
// ---------------------------------------------------------------------------

function bruteForceTopK(
  nodes: Map<string, { vector: number[] }>,
  query: number[],
  k: number
): string[] {
  const sims: Array<{ key: string; sim: number }> = []
  for (const [key, node] of nodes) {
    let dot = 0
    let magA = 0
    let magB = 0
    for (let i = 0; i < query.length; i++) {
      dot += node.vector[i] * query[i]
      magA += node.vector[i] * node.vector[i]
      magB += query[i] * query[i]
    }
    const denom = Math.sqrt(magA) * Math.sqrt(magB)
    sims.push({ key, sim: denom === 0 ? 0 : dot / denom })
  }
  sims.sort((a, b) => b.sim - a.sim)
  return sims.slice(0, k).map((e) => e.key)
}

function randomUnitVector(dim: number): number[] {
  const v = Array.from({ length: dim }, () => Math.random() * 2 - 1)
  const mag = Math.sqrt(v.reduce((s, x) => s + x * x, 0))
  return v.map((x) => x / mag)
}

// ---------------------------------------------------------------------------
// Group 1: Construction
// ---------------------------------------------------------------------------

describe('HnswIndex construction', () => {
  it('stores default params correctly', () => {
    const idx = new TestableHnswIndex()
    expect((idx as unknown as { maxLayer: number }).maxLayer).toBe(3)
    expect(
      (idx as unknown as { maxNumberOfNeighbors: number }).maxNumberOfNeighbors
    ).toBe(10)
    expect((idx as unknown as { efConstruction: number }).efConstruction).toBe(
      200
    )
    expect((idx as unknown as { efSearch: number }).efSearch).toBe(50)
  })

  it('stores custom params correctly', () => {
    const idx = new TestableHnswIndex(5, 20, 100, 30)
    expect((idx as unknown as { maxLayer: number }).maxLayer).toBe(5)
    expect(
      (idx as unknown as { maxNumberOfNeighbors: number }).maxNumberOfNeighbors
    ).toBe(20)
    expect((idx as unknown as { efConstruction: number }).efConstruction).toBe(
      100
    )
    expect((idx as unknown as { efSearch: number }).efSearch).toBe(30)
  })
})

// ---------------------------------------------------------------------------
// Group 2: has()
// ---------------------------------------------------------------------------

describe('HnswIndex has()', () => {
  it('returns false on empty index', () => {
    const idx = new HnswIndex()
    expect(idx.has('missing')).toBe(false)
  })

  it('returns true after insert, false for other keys', () => {
    const idx = new HnswIndex()
    idx.insert('a', [1, 0])
    expect(idx.has('a')).toBe(true)
    expect(idx.has('b')).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// Group 3: insert internals (via TestableHnswIndex)
// ---------------------------------------------------------------------------

describe('HnswIndex insert internals', () => {
  it('adds key to nodes Map', () => {
    const idx = new TestableHnswIndex()
    idx.insert('x', [1, 0, 0])
    expect(idx.getNodes().has('x')).toBe(true)
  })

  it('assigns layer in [0, maxLayer]; higher layers are rarer (1000 samples)', () => {
    const idx = new TestableHnswIndex(3, 10, 200, 50)
    const counts = [0, 0, 0, 0] // layers 0..3
    for (let i = 0; i < 1000; i++) {
      const key = `k${i}`
      idx.insert(key, randomUnitVector(8))
      const node = idx.getNodes().get(key)
      if (node) {
        counts[node.maxLayer]++
      }
    }
    // Layer 0 should have the most nodes; each higher layer should have fewer
    expect(counts[0]).toBeGreaterThan(counts[1])
    expect(counts[1]).toBeGreaterThan(0)
  })

  it('sets entrypoint to first inserted key', () => {
    const idx = new TestableHnswIndex()
    idx.insert('first', [1, 0])
    expect(idx.getEntrypointKey()).toBe('first')
  })

  it('does not replace entrypoint when new node has lower or equal layer', () => {
    // Force layers via a subclass that overrides assignLayer
    class LayeredIndex extends TestableHnswIndex {
      private layerSeq: number[]
      private layerPos = 0
      constructor(layers: number[]) {
        super(3, 10, 200, 50)
        this.layerSeq = layers
      }
      protected assignLayer(): number {
        return this.layerSeq[this.layerPos++ % this.layerSeq.length]
      }
    }

    const idx = new LayeredIndex([2, 1]) // first at layer 2, second at layer 1
    idx.insert('high', [1, 0])
    idx.insert('low', [0, 1])
    expect(idx.getEntrypointKey()).toBe('high')
  })

  it('updates entrypoint when new node has higher layer', () => {
    class LayeredIndex extends TestableHnswIndex {
      private layerSeq: number[]
      private layerPos = 0
      constructor(layers: number[]) {
        super(3, 10, 200, 50)
        this.layerSeq = layers
      }
      protected assignLayer(): number {
        return this.layerSeq[this.layerPos++ % this.layerSeq.length]
      }
    }

    const idx = new LayeredIndex([1, 3]) // first at layer 1, second at layer 3
    idx.insert('low', [1, 0])
    idx.insert('high', [0, 1])
    expect(idx.getEntrypointKey()).toBe('high')
  })

  it('creates bidirectional neighbor links at layer 0', () => {
    const idx = new TestableHnswIndex(3, 10, 200, 50)
    idx.insert('a', [1, 0])
    idx.insert('b', [0.9, 0.1])

    const nodes = idx.getNodes()
    const aNeighbors = nodes.get('a')?.neighbors.get(0) ?? []
    const bNeighbors = nodes.get('b')?.neighbors.get(0) ?? []

    expect(aNeighbors).toContain('b')
    expect(bNeighbors).toContain('a')
  })

  it('no neighbor list exceeds maxNumberOfNeighbors after many inserts', () => {
    const m = 5
    const idx = new TestableHnswIndex(3, m, 200, 50)
    for (let i = 0; i < 100; i++) {
      idx.insert(`node${i}`, randomUnitVector(8))
    }
    for (const [, node] of idx.getNodes()) {
      for (const [, neighbors] of node.neighbors) {
        expect(neighbors.length).toBeLessThanOrEqual(m)
      }
    }
  })
})

// ---------------------------------------------------------------------------
// Group 4: search basics
// ---------------------------------------------------------------------------

describe('HnswIndex search basics', () => {
  it('returns [] on empty index', () => {
    const idx = new HnswIndex()
    expect(idx.search([1, 0], 5)).toEqual([])
  })

  it('returns [] when k = 0', () => {
    const idx = new HnswIndex()
    idx.insert('a', [1, 0])
    expect(idx.search([1, 0], 0)).toEqual([])
  })

  it('returns the single inserted key', () => {
    const idx = new HnswIndex()
    idx.insert('only', [1, 0, 0])
    const results = idx.search([1, 0, 0], 1)
    expect(results).toEqual(['only'])
  })

  it('returns more-similar key first', () => {
    const idx = new HnswIndex()
    // 'close' is nearly identical to the query; 'far' is orthogonal
    idx.insert('close', [1, 0])
    idx.insert('far', [0, 1])
    const results = idx.search([1, 0], 2)
    expect(results[0]).toBe('close')
  })

  it('returns exactly k keys when k < n', () => {
    const idx = new HnswIndex()
    for (let i = 0; i < 10; i++) {
      idx.insert(`n${i}`, randomUnitVector(8))
    }
    const results = idx.search(randomUnitVector(8), 3)
    expect(results).toHaveLength(3)
  })

  it('returns results sorted by similarity descending', () => {
    const idx = new HnswIndex()
    idx.insert('a', [1, 0, 0])
    idx.insert('b', [0.7, 0.7, 0])
    idx.insert('c', [0, 1, 0])

    const query = [1, 0, 0]
    const results = idx.search(query, 3)

    // Verify sorted order by recomputing similarities
    function sim(v: number[]): number {
      let dot = 0
      let ma = 0
      let mb = 0
      for (let i = 0; i < v.length; i++) {
        dot += v[i] * query[i]
        ma += v[i] * v[i]
        mb += query[i] * query[i]
      }
      return dot / (Math.sqrt(ma) * Math.sqrt(mb))
    }

    const vectors: Record<string, number[]> = {
      a: [1, 0, 0],
      b: [0.7, 0.7, 0],
      c: [0, 1, 0]
    }
    for (let i = 0; i < results.length - 1; i++) {
      expect(sim(vectors[results[i]])).toBeGreaterThanOrEqual(
        sim(vectors[results[i + 1]])
      )
    }
  })

  it('returns all n keys when k > n', () => {
    const idx = new HnswIndex()
    idx.insert('x', [1, 0])
    idx.insert('y', [0, 1])
    const results = idx.search([1, 0], 10)
    expect(results).toHaveLength(2)
    expect(results).toContain('x')
    expect(results).toContain('y')
  })
})

// ---------------------------------------------------------------------------
// Group 5: Search accuracy
// ---------------------------------------------------------------------------

describe('HnswIndex search accuracy', () => {
  it('1-NN recall >= 90% on 100 nodes, 8 dims (20 queries)', () => {
    const idx = new TestableHnswIndex(4, 16, 200, 100)
    const nodeMap: Map<string, { vector: number[] }> = new Map()

    for (let i = 0; i < 100; i++) {
      const v = randomUnitVector(8)
      idx.insert(`n${i}`, v)
      nodeMap.set(`n${i}`, { vector: v })
    }

    let hits = 0
    for (let q = 0; q < 20; q++) {
      const query = randomUnitVector(8)
      const hnswResult = idx.search(query, 1)
      const exactResult = bruteForceTopK(nodeMap, query, 1)
      if (hnswResult[0] === exactResult[0]) {
        hits++
      }
    }

    expect(hits / 20).toBeGreaterThanOrEqual(0.9)
  })

  it('top-10 recall >= 80% on 500 nodes, 16 dims (50 queries)', () => {
    const idx = new TestableHnswIndex(4, 16, 200, 100)
    const nodeMap: Map<string, { vector: number[] }> = new Map()

    for (let i = 0; i < 500; i++) {
      const v = randomUnitVector(16)
      idx.insert(`n${i}`, v)
      nodeMap.set(`n${i}`, { vector: v })
    }

    let totalRecall = 0
    for (let q = 0; q < 50; q++) {
      const query = randomUnitVector(16)
      const hnswResult = idx.search(query, 10)
      const exactResult = bruteForceTopK(nodeMap, query, 10)

      const exactSet = new Set(exactResult)
      const overlap = hnswResult.filter((k) => exactSet.has(k)).length
      totalRecall += overlap / 10
    }

    expect(totalRecall / 50).toBeGreaterThanOrEqual(0.8)
  })
})

// ---------------------------------------------------------------------------
// Group 6: delete
// ---------------------------------------------------------------------------

describe('HnswIndex delete', () => {
  it('returns false for nonexistent key', () => {
    const idx = new HnswIndex()
    expect(idx.delete('ghost')).toBe(false)
  })

  it('returns true and has() returns false after delete', () => {
    const idx = new HnswIndex()
    idx.insert('a', [1, 0])
    expect(idx.delete('a')).toBe(true)
    expect(idx.has('a')).toBe(false)
  })

  it('deleted key absent from search results', () => {
    const idx = new HnswIndex()
    idx.insert('a', [1, 0])
    idx.insert('b', [0.9, 0.1])
    idx.delete('a')
    const results = idx.search([1, 0], 5)
    expect(results).not.toContain('a')
  })

  it('deleted key absent from all neighbor lists', () => {
    const idx = new TestableHnswIndex()
    for (let i = 0; i < 10; i++) {
      idx.insert(`n${i}`, randomUnitVector(4))
    }
    idx.delete('n0')
    for (const [, node] of idx.getNodes()) {
      for (const [, neighbors] of node.neighbors) {
        expect(neighbors).not.toContain('n0')
      }
    }
  })

  it('deleting entrypoint sets new valid entrypoint; search still works', () => {
    const idx = new TestableHnswIndex()
    idx.insert('a', [1, 0])
    idx.insert('b', [0, 1])
    const ep = idx.getEntrypointKey()
    if (ep === null) {
      throw new Error('Expected entrypoint to be set after insert')
    }
    idx.delete(ep)
    expect(idx.getEntrypointKey()).not.toBeNull()
    const results = idx.search([1, 0], 1)
    expect(results).toHaveLength(1)
  })

  it('deleting last node sets entrypointKey to null; search returns []', () => {
    const idx = new TestableHnswIndex()
    idx.insert('only', [1, 0])
    idx.delete('only')
    expect(idx.getEntrypointKey()).toBeNull()
    expect(idx.search([1, 0], 5)).toEqual([])
  })

  it('deleting all nodes leaves clean state', () => {
    const idx = new TestableHnswIndex()
    for (let i = 0; i < 5; i++) {
      idx.insert(`n${i}`, randomUnitVector(4))
    }
    for (let i = 0; i < 5; i++) {
      idx.delete(`n${i}`)
    }
    expect(idx.getNodes().size).toBe(0)
    expect(idx.getEntrypointKey()).toBeNull()
    expect(idx.search(randomUnitVector(4), 5)).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// Group 7: Upsert
// ---------------------------------------------------------------------------

describe('HnswIndex upsert', () => {
  it('re-inserting existing key replaces vector and re-links', () => {
    const idx = new TestableHnswIndex()
    idx.insert('x', [1, 0, 0])
    idx.insert('y', [0, 1, 0])

    // Update x to point toward y's direction
    idx.insert('x', [0, 1, 0])

    const nodes = idx.getNodes()
    expect(nodes.size).toBe(2)
    expect(nodes.get('x')?.vector).toEqual([0, 1, 0])

    // x should not be its own neighbor
    const xNode = nodes.get('x')
    if (xNode) {
      for (const [, neighbors] of xNode.neighbors) {
        expect(neighbors).not.toContain('x')
      }
    }

    // Search should find x with its new vector (both x and y are [0,1,0] so check both appear)
    const results = idx.search([0, 1, 0], 2)
    expect(results).toContain('x')
  })
})
