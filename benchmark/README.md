# Olingo Benchmark Suite

Performance benchmarks for the Olingo embedding database.

## Quick Start

```bash
# Quick benchmark (smaller dataset sizes)
bun run benchmark:quick

# Full benchmark (includes 5k and 10k documents if available)
bun run benchmark
```

## What It Tests

### 1. **Store Operations**

Measures bulk insert performance using `storeMany()`.

**Metrics:**

- Total duration
- Throughput (documents/second)
- Average time per document

**Dataset sizes:** 100, 500, 1K, 2K documents

### 2. **Search Operations**

Measures semantic search performance with cold start and warm queries.

**Metrics:**

- Cold start latency (first query)
- Average warm query latency

**Dataset sizes:** 100, 500, 1K, 2K, 5K, 10K documents

### 3. **Get Operations**

Measures key-based retrieval performance.

**Metrics:**

- Sequential access average latency
- Random access average latency

**Dataset sizes:** 100, 500, 1K, 2K, 5K, 10K documents

### 4. **Scalability Analysis**

Analyzes how performance scales with dataset size.

**Metrics:**

- Size ratio (e.g., 2x, 5x)
- Time ratio
- Efficiency percentage

## Sample Data

The benchmark uses JSON article data with the following structure:

```json
{
  "title": "Article title",
  "text": "Article content...",
  "categories": ["Category1", "Category2"],
  "topics": ["topic1", "topic2"],
  "sentiment": "positive"
}
```

Currently loaded: **2,000 articles** from `benchmark/data`

### Adding More Data for 5K/10K Benchmarks

To test with 5,000 or 10,000 documents:

1. Download additional article JSON files
2. Place them in subdirectories under `benchmark/data/`
3. Each subdirectory should contain article JSON files named `article_*.json`

Example structure:

```
benchmark/data/
├── dataset_1/
│   ├── article_1.json
│   ├── article_2.json
│   └── ...
├── dataset_2/
│   ├── article_1.json
│   └── ...
└── dataset_3/
    └── ...
```

The benchmark automatically loads from all subdirectories until it reaches the
required number of documents.

## Custom Options

```bash
# Specify custom data directory
bun run benchmark/index.ts --data-dir=/path/to/data

# Quick mode
bun run benchmark/index.ts --quick
```

## Output

The benchmark produces GitHub-style markdown tables showing:

- **Store Operations**: Duration, throughput, avg/doc
- **Search Operations**: Cold start, avg warm latency
- **Get Operations**: Sequential vs random access times
- **Scalability Analysis**: Efficiency ratios across sizes
- **Summary**: Key insights and metrics

All tables are formatted with pipe delimiters and can be directly copied into
documentation.

## Test Queries

The benchmark uses 20 predefined queries covering different topics:

- Cinema and awards
- Sports and athletics
- Technology and software
- Politics and government
- Business and finance
- Health and medicine
- Environment and climate
- Education and learning
- Travel and tourism
- Food and cooking
- Celebrity and entertainment
- Crime and law
- Science and research
- Fashion and style
- Music and concerts
- Real estate and housing
- Automotive and vehicles
- Social media and trends
- Weather and disasters
- Books and literature

## Implementation

The benchmark suite consists of:

- `benchmark/index.ts` - Main orchestrator
- `benchmark/suites/` - Individual benchmark implementations
  - `store-benchmark.ts` - Bulk insert tests
  - `search-benchmark.ts` - Search performance tests
  - `get-benchmark.ts` - Key retrieval tests
- `benchmark/utils/` - Helpers for data loading and reporting
- `benchmark/queries.ts` - Test query definitions

### Timing Methodology

All benchmarks use **total time measurement** rather than individual operation
timing to:

- Reduce measurement overhead from repeated `performance.now()` calls
- Provide more stable, less noisy results
- Better reflect real-world batch usage patterns
- Calculate accurate throughput metrics

## Performance Tips

1. **Store operations** are slower due to embedding generation - keep dataset
   sizes reasonable (≤2K)
2. **Search and Get** operations can handle much larger datasets (5K-10K)
3. The benchmark uses `storeMany()` for efficient bulk inserts
4. Databases are created once and reused for read benchmarks
5. Temporary files are cleaned up automatically after completion
