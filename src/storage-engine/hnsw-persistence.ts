/**
 * Serialization and deserialization for HNSW indexes.
 *
 * Format:
 * JSON-encoded structure containing:
 * - config: { maxLayer, maxNumberOfNeighbors, efConstruction, efSearch }
 * - entrypointKey
 * - nodes: Array of { key, vector, maxLayer, neighbors: Record<layer, keys[]> }
 */

import { HnswIndex, Node } from './hnsw-index'

interface SerializedNode {
  key: string
  vector: number[]
  maxLayer: number
  neighbors: Record<string, string[]>
}

interface SerializedHnswIndex {
  version: 1
  config: {
    maxLayer: number
    maxNumberOfNeighbors: number
    efConstruction: number
    efSearch: number
  }
  entrypointKey: string | null
  nodes: SerializedNode[]
}

/**
 * Serialize an HNSW index to a Uint8Array.
 */
export function serializeHnswIndex(index: HnswIndex): Uint8Array {
  const data: SerializedHnswIndex = {
    version: 1,
    config: {
      maxLayer: index['maxLayer'],
      maxNumberOfNeighbors: index['maxNumberOfNeighbors'],
      efConstruction: index['efConstruction'],
      efSearch: index['efSearch']
    },
    entrypointKey: index['entrypointKey'],
    nodes: []
  }

  for (const [, node] of index['nodes']) {
    const neighbors: Record<string, string[]> = {}
    for (const [layer, keys] of node.neighbors) {
      neighbors[String(layer)] = keys
    }
    data.nodes.push({
      key: node.key,
      vector: node.vector,
      maxLayer: node.maxLayer,
      neighbors
    })
  }

  const json = JSON.stringify(data)
  return new TextEncoder().encode(json)
}

/**
 * Deserialize an HNSW index from a Uint8Array.
 */
export function deserializeHnswIndex(bytes: Uint8Array): HnswIndex {
  const json = new TextDecoder().decode(bytes)
  const data = JSON.parse(json) as SerializedHnswIndex

  const index = new HnswIndex(
    data.config.maxLayer,
    data.config.maxNumberOfNeighbors,
    data.config.efConstruction,
    data.config.efSearch
  )

  // Reconstruct nodes directly
  const nodesMap = index['nodes']

  for (const serializedNode of data.nodes) {
    const node = new Node(
      serializedNode.key,
      serializedNode.vector,
      serializedNode.maxLayer
    )
    for (const [layerStr, keys] of Object.entries(serializedNode.neighbors)) {
      node.neighbors.set(Number(layerStr), keys)
    }
    nodesMap.set(serializedNode.key, node)
  }

  // Restore entrypoint
  ;(index as unknown as { entrypointKey: string | null }).entrypointKey =
    data.entrypointKey

  return index
}
