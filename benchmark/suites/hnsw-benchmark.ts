import { HnswIndex } from '../../src/storage-engine/hnsw-index'
import {
  formatDuration,
  formatPercentage,
  printMarkdownTable,
  printModernSection
} from '../utils/reporter'

const DIMS = 384
const K = 10
const NUM_QUERIES = 20
const DATASET_SIZES = [1000, 5000, 10000, 50000]

function randomUnitVector(dims: number): number[] {
  const v = Array.from({ length: dims }, () => Math.random() * 2 - 1)
  const mag = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0))
  return v.map((x) => x / mag)
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
  }
  // Unit vectors: dot product equals cosine similarity
  return dot
}

function linearSearch(
  vectors: number[][],
  query: number[],
  k: number
): number[] {
  const scored = vectors.map((v, i) => ({ i, sim: cosineSimilarity(v, query) }))
  scored.sort((a, b) => b.sim - a.sim)
  return scored.slice(0, k).map((e) => e.i)
}

interface BenchmarkResult {
  size: number
  linearAvgMs: number
  buildMs: number
  hnswAvgMs: number
  recall: number
  speedup: number
}

async function runForSize(n: number): Promise<BenchmarkResult> {
  // Generate dataset
  const vectors: number[][] = Array.from({ length: n }, () =>
    randomUnitVector(DIMS)
  )
  const queries: number[][] = Array.from({ length: NUM_QUERIES }, () =>
    randomUnitVector(DIMS)
  )

  // --- Linear scan baseline ---
  const linearStart = performance.now()
  const groundTruth: number[][] = []
  for (const query of queries) {
    groundTruth.push(linearSearch(vectors, query, K))
  }
  const linearMs = performance.now() - linearStart
  const linearAvgMs = linearMs / NUM_QUERIES

  // --- Build HNSW index ---
  const buildStart = performance.now()
  const index = new HnswIndex(3, 10, 200, 50)
  for (let i = 0; i < n; i++) {
    index.insert(String(i), vectors[i])
  }
  const buildMs = performance.now() - buildStart

  // --- HNSW search ---
  const hnswStart = performance.now()
  const hnswResults: number[][] = []
  for (const query of queries) {
    const keys = index.search(query, K)
    hnswResults.push(keys.map((k) => parseInt(k, 10)))
  }
  const hnswMs = performance.now() - hnswStart
  const hnswAvgMs = hnswMs / NUM_QUERIES

  // --- Recall@K ---
  let totalIntersection = 0
  for (let q = 0; q < NUM_QUERIES; q++) {
    const gt = new Set(groundTruth[q])
    const found = hnswResults[q].filter((id) => gt.has(id)).length
    totalIntersection += found
  }
  const recall = totalIntersection / (NUM_QUERIES * K)

  return {
    size: n,
    linearAvgMs,
    buildMs,
    hnswAvgMs,
    recall,
    speedup: linearAvgMs / hnswAvgMs
  }
}

async function main(): Promise<void> {
  printModernSection('HNSW vs Linear Scan Benchmark')
  console.log(
    `Synthetic random unit vectors — ${DIMS} dims, top-${K}, ${NUM_QUERIES} queries per size`
  )
  console.log()

  const rows = []

  for (const size of DATASET_SIZES) {
    process.stdout.write(`Running N=${size.toLocaleString()}...`)
    const result = await runForSize(size)
    process.stdout.write(' done\n')

    rows.push({
      N: result.size.toLocaleString(),
      'Linear avg': formatDuration(result.linearAvgMs),
      'Build time': formatDuration(result.buildMs),
      'HNSW avg': formatDuration(result.hnswAvgMs),
      Speedup: `${result.speedup.toFixed(1)}x`,
      'Recall@10': formatPercentage(result.recall)
    })
  }

  console.log()
  printModernSection('Results')

  printMarkdownTable(
    [
      { header: 'N', align: 'right' },
      { header: 'Linear avg', align: 'right' },
      { header: 'Build time', align: 'right' },
      { header: 'HNSW avg', align: 'right' },
      { header: 'Speedup', align: 'right' },
      { header: 'Recall@10', align: 'right' }
    ],
    rows
  )
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
