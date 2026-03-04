import invariant from 'tiny-invariant'

export class HnswIndex {
  protected readonly maxLayer: number
  protected readonly maxNumberOfNeighbors: number
  protected readonly efConstruction: number
  protected readonly efSearch: number
  protected readonly nodes: Map<string, Node> = new Map()

  protected entrypointKey: string | null = null

  constructor(
    maxLayer = 3,
    maxNumberOfNeighbors = 10,
    efConstruction = 200,
    efSearch = 50
  ) {
    this.maxLayer = maxLayer
    this.maxNumberOfNeighbors = maxNumberOfNeighbors
    this.efConstruction = efConstruction
    this.efSearch = efSearch
  }

  has(key: string): boolean {
    return this.nodes.has(key)
  }

  getVector(key: string): number[] | null {
    return this.nodes.get(key)?.vector ?? null
  }

  insert(key: string, vector: number[]): void {
    invariant(key, 'Key must be a non-empty string')
    invariant(
      vector && vector.length > 0,
      'Vector must be a non-empty array of numbers'
    )

    // Upsert: remove stale entry first
    if (this.nodes.has(key)) {
      this.delete(key)
    }

    const nodeLevel = this.assignLayer()
    const node = new Node(key, vector, nodeLevel)

    this.nodes.set(key, node)

    if (this.entrypointKey === null) {
      this.entrypointKey = key

      return
    }

    const entrypointNode = this.nodes.get(this.entrypointKey)

    invariant(
      entrypointNode,
      `Entrypoint key "${this.entrypointKey}" not found in nodes`
    )

    let ep: string[] = [this.entrypointKey]
    const topLayer = entrypointNode.maxLayer

    // Greedy descent from topLayer down to nodeLevel+1 (ef=1)
    for (let l = topLayer; l >= nodeLevel + 1; l--) {
      ep = this.searchLayer(ep, vector, 1, l).slice(0, 1)
    }

    // Beam search and link from min(nodeLevel, topLayer) down to 0
    for (let l = Math.min(nodeLevel, topLayer); l >= 0; l--) {
      const candidates = this.searchLayer(ep, vector, this.efConstruction, l)
      const neighbors = this.selectNeighbors(
        candidates,
        this.maxNumberOfNeighbors
      )

      node.neighbors.set(l, [...neighbors])

      for (const neighborKey of neighbors) {
        const neighborNode = this.nodes.get(neighborKey)

        if (!neighborNode) {
          continue
        }

        const neighborList = neighborNode.neighbors.get(l) ?? []

        neighborList.push(key)
        neighborNode.neighbors.set(l, neighborList)

        if (neighborList.length > this.maxNumberOfNeighbors) {
          this.shrinkNeighbors(neighborKey, l, this.maxNumberOfNeighbors)
        }
      }

      ep = candidates
    }

    if (nodeLevel > entrypointNode.maxLayer) {
      this.entrypointKey = key
    }
  }

  search(query: number[], k: number): string[] {
    invariant(
      query && query.length > 0,
      'Query vector must be a non-empty array of numbers'
    )
    invariant(k >= 0, 'k must be a non-negative integer')

    if (this.entrypointKey === null || k === 0) {
      return []
    }

    const entrypointNode = this.nodes.get(this.entrypointKey)

    if (!entrypointNode) {
      return []
    }

    let ep: string[] = [this.entrypointKey]
    const topLayer = entrypointNode.maxLayer

    // Greedy descent layers topLayer → 1
    for (let l = topLayer; l >= 1; l--) {
      ep = this.searchLayer(ep, query, 1, l).slice(0, 1)
    }

    // Full beam search at layer 0
    const results = this.searchLayer(ep, query, this.efSearch, 0)

    return results.slice(0, k)
  }

  delete(key: string): boolean {
    invariant(key, 'Key must be a non-empty string')

    if (!this.nodes.has(key)) {
      return false
    }

    const node = this.nodes.get(key)

    if (!node) {
      return false
    }

    // Remove key from all neighbor lists
    for (let l = 0; l <= node.maxLayer; l++) {
      const neighbors = node.neighbors.get(l) ?? []

      for (const neighborKey of neighbors) {
        const neighborNode = this.nodes.get(neighborKey)

        if (!neighborNode) {
          continue
        }

        const neighborList = neighborNode.neighbors.get(l) ?? []
        const filtered = neighborList.filter((k) => k !== key)

        neighborNode.neighbors.set(l, filtered)
      }
    }

    this.nodes.delete(key)

    if (key === this.entrypointKey) {
      this.entrypointKey = this.findNewEntrypoint()
    }

    return true
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0
    let magA = 0
    let magB = 0
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i]
      magA += a[i] * a[i]
      magB += b[i] * b[i]
    }
    const denom = Math.sqrt(magA) * Math.sqrt(magB)
    if (denom === 0) {
      return 0
    }
    return dot / denom
  }

  protected assignLayer(): number {
    // Exponential decay: floor(-ln(U) * 1/ln(m)), clamped to [0, maxLayer]
    const m = this.maxNumberOfNeighbors
    const u = Math.random()
    const level = Math.floor(-Math.log(u) * (1 / Math.log(m)))
    return Math.min(level, this.maxLayer)
  }

  private searchLayer(
    entryNodes: string[],
    queryVector: number[],
    ef: number,
    layer: number
  ): string[] {
    const visited = new Set<string>(entryNodes)

    // Build initial entries with similarities
    const initial: Array<{ key: string; sim: number }> = []
    for (const key of entryNodes) {
      const node = this.nodes.get(key)
      if (!node) {
        continue
      }
      const sim = this.cosineSimilarity(node.vector, queryVector)
      initial.push({ key, sim })
    }

    // W: result set, sorted descending by sim
    const W: Array<{ key: string; sim: number }> = [...initial].sort(
      (a, b) => b.sim - a.sim
    )
    // C: frontier, sorted ascending by sim (pop() gives best)
    const C: Array<{ key: string; sim: number }> = [...initial].sort(
      (a, b) => a.sim - b.sim
    )

    while (C.length > 0) {
      const c = C.pop()
      if (!c) {
        break
      }

      // If W is full and c is worse than worst in W, stop
      if (W.length >= ef && c.sim < W[W.length - 1].sim) {
        break
      }

      const cNode = this.nodes.get(c.key)
      if (!cNode) {
        continue
      }

      const layerNeighbors = cNode.neighbors.get(layer) ?? []
      for (const neighborKey of layerNeighbors) {
        if (visited.has(neighborKey)) {
          continue
        }
        visited.add(neighborKey)

        const neighborNode = this.nodes.get(neighborKey)
        if (!neighborNode) {
          continue
        }

        const eSim = this.cosineSimilarity(neighborNode.vector, queryVector)
        const worstW = W.length > 0 ? W[W.length - 1].sim : -Infinity

        if (W.length < ef || eSim > worstW) {
          // Insert into C (maintain ascending order)
          insertSorted(
            C,
            { key: neighborKey, sim: eSim },
            (a, b) => a.sim - b.sim
          )
          // Insert into W (maintain descending order, trim to ef)
          insertSorted(
            W,
            { key: neighborKey, sim: eSim },
            (a, b) => b.sim - a.sim
          )
          if (W.length > ef) {
            W.pop()
          }
        }
      }
    }

    return W.map((entry) => entry.key)
  }

  private selectNeighbors(candidates: string[], m: number): string[] {
    // Simple heuristic: candidates already sorted desc by sim, take best m
    return candidates.slice(0, m)
  }

  private shrinkNeighbors(nodeKey: string, layer: number, m: number): void {
    const node = this.nodes.get(nodeKey)
    if (!node) {
      return
    }
    const neighbors = node.neighbors.get(layer) ?? []
    if (neighbors.length <= m) {
      return
    }

    // Sort by similarity to the node and keep top m
    const nodeVector = node.vector
    const withSim = neighbors
      .map((k) => {
        const n = this.nodes.get(k)
        if (!n) {
          return { key: k, sim: -Infinity }
        }
        return { key: k, sim: this.cosineSimilarity(nodeVector, n.vector) }
      })
      .sort((a, b) => b.sim - a.sim)
      .slice(0, m)

    node.neighbors.set(
      layer,
      withSim.map((e) => e.key)
    )
  }

  private findNewEntrypoint(): string | null {
    let bestKey: string | null = null
    let bestLayer = -1
    for (const [key, node] of this.nodes) {
      if (node.maxLayer > bestLayer) {
        bestLayer = node.maxLayer
        bestKey = key
      }
    }
    return bestKey
  }
}

// ---------------------------------------------------------------------------
// Insertion helper: insert item into sorted array maintaining given comparator
// ---------------------------------------------------------------------------

function insertSorted<T>(
  arr: T[],
  item: T,
  comparator: (a: T, b: T) => number
): void {
  let lo = 0
  let hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (comparator(arr[mid], item) <= 0) {
      lo = mid + 1
    } else {
      hi = mid
    }
  }
  arr.splice(lo, 0, item)
}

// ---------------------------------------------------------------------------
// Node class
// ---------------------------------------------------------------------------

export class Node {
  public readonly key: string
  public maxLayer: number
  public readonly neighbors: Map<number, string[]> = new Map()
  public readonly vector: number[]

  constructor(key: string, vector: number[], maxLayer: number) {
    this.key = key
    this.vector = vector
    this.maxLayer = maxLayer
  }
}
