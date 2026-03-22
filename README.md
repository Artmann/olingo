# Olingo

[![CI](https://img.shields.io/github/actions/workflow/status/artmann/olingo/ci.yml?branch=main&label=CI&logo=github)](https://github.com/artmann/olingo/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/olingo.svg?logo=npm)](https://www.npmjs.com/package/olingo)
[![npm downloads](https://img.shields.io/npm/dm/olingo.svg)](https://www.npmjs.com/package/olingo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg?logo=typescript)](https://www.typescriptlang.org/)

> A lightweight semantic search database with text embeddings for Node.js and
> Bun

Olingo lets you build semantic search into your applications with just a few
lines of code. Store text, search by meaning, and find similar content—perfect
for RAG systems, chatbots, and recommendation engines.

## What is Olingo?

Olingo is an embedding database that automatically converts text into vector
embeddings and stores them in an efficient binary format. Instead of searching
by exact keywords, you can search by semantic similarity—finding documents that
mean the same thing, even if they use different words.

**Example:** Search for "how to reset password" and find results like "forgot my
login credentials" or "change account password".

## Why Olingo?

- **Simple API** - No complex setup, just store and search
- **Semantic Search** - Find content by meaning, not just keywords
- **Fast & Efficient** - Binary storage format ~50% smaller than JSON
- **Zero Dependencies** - Embeddings generated locally, no API keys needed
- **Works Everywhere** - Compatible with Node.js and Bun
- **Built for RAG** - Perfect for Retrieval Augmented Generation systems

## Use Cases

- **FAQ Bots** - Match user questions to answers by meaning
- **Document Search** - Semantic search over documentation or knowledge bases
- **Code Search** - Find code snippets by describing what they do
- **Content Recommendations** - "More like this" functionality
- **RAG Systems** - Retrieve relevant context for LLM prompts

## Installation

```bash
# Using npm
npm install olingo

# Using bun
bun add olingo
```

## Quick Start

### Programmatic API

```typescript
import { EmbeddingEngine } from 'olingo'

const engine = new EmbeddingEngine({
  storePath: './my-database.raptor'
})

// Store documents
await engine.storeMany([
  { key: 'doc1', text: 'How to reset your password' },
  { key: 'doc2', text: 'Machine learning basics' },
  { key: 'doc3', text: 'Getting started with Bun' }
])

// Search by meaning
const results = await engine.search('forgot my password', 5)
console.log(results[0].key) // 'doc1' - matched by meaning!
console.log(results[0].similarity) // 0.87 - high similarity score
```

### Command Line Interface

```bash
# Store documents
olingo store doc1 "How to reset your password"
olingo store doc2 "Machine learning basics"

# Search by meaning
olingo search "forgot my password" --limit 5

# Retrieve by key
olingo get doc1
```

## Examples

See the [examples/](examples/) directory for complete, runnable examples:

| Example                    | Description                                             | Run                                             |
| -------------------------- | ------------------------------------------------------- | ----------------------------------------------- |
| **Document Search / RAG**  | Semantic search over documentation chunks               | `bun run examples/01-document-search.ts`        |
| **FAQ Bot**                | Match user questions to FAQs with confidence thresholds | `bun run examples/02-faq-bot.ts`                |
| **Code Snippet Library**   | Search code by natural language descriptions            | `bun run examples/03-code-snippets.ts`          |
| **Content Recommendation** | "More like this" functionality                          | `bun run examples/04-content-recommendation.ts` |

## API Reference

### `EmbeddingEngine`

#### `constructor(options)`

Create a new embedding engine.

```typescript
const engine = new EmbeddingEngine({
  storePath: './database.raptor', // Path to storage file
  embeddingCacheSize: 100 // Optional: cache up to 100 text-to-embedding mappings
})
```

**Options:**

- `storePath` - Path to the database file (required)
- `cacheDir` - Directory to cache downloaded models (default: `./.cache/models`)
- `readOnly` - Open database in read-only mode (default: `false`)
- `embeddingCacheSize` - Size of the LRU cache for text-to-embedding lookups
  (default: `0` = disabled)
- `embeddingProvider` - Custom embedding provider (optional). When provided, the
  default model is not loaded. See
  [Custom Embedding Providers](#custom-embedding-providers).

#### `store(key, text)`

Store a single text entry with auto-generated embedding.

```typescript
await engine.store('doc1', 'The quick brown fox')
```

#### `storeMany(items)`

Store multiple entries in batch (faster than multiple `store()` calls).

```typescript
await engine.storeMany([
  { key: 'doc1', text: 'First document' },
  { key: 'doc2', text: 'Second document' }
])
```

#### `search(query, limit?, minSimilarity?)` / `search(query, options?)`

Search for semantically similar entries. Supports both positional arguments and
a `SearchOptions` object.

```typescript
// Positional arguments (backward compatible)
const results = await engine.search('artificial intelligence', 10, 0.7)

// SearchOptions object
const results2 = await engine.search('artificial intelligence', {
  limit: 10,
  minSimilarity: 0.7
})

// Detailed results with similarity breakdown
const detailed = await engine.search('artificial intelligence', {
  limit: 10,
  minSimilarity: 0.7,
  includeDetails: true
})
detailed.forEach((r) => {
  console.log(r.key, r.similarity)
  console.log(r.queryNorm, r.resultNorm, r.dotProduct)
})
```

#### `update(key, text)`

Update the text for an existing key. Throws `KeyNotFoundError` if the key
doesn't exist.

```typescript
await engine.update('doc1', 'new text for doc1')
```

#### `storeEmbedding(key, embedding)`

Store a pre-computed embedding directly, bypassing the embedding model.

```typescript
const embedding = new Float32Array(384) // or number[]
await engine.storeEmbedding('doc1', embedding)
```

#### `storeManyEmbeddings(items)`

Store multiple pre-computed embeddings in batch.

```typescript
await engine.storeManyEmbeddings([
  { key: 'doc1', embedding: new Float32Array(384) },
  { key: 'doc2', embedding: [0.1, 0.2 /* ... */] }
])
```

#### `searchMany(queries, limit?, minSimilarity?)`

Search for multiple queries in batch. Deduplicates identical queries and
generates embeddings in parallel.

```typescript
const results = await engine.searchMany(['hello', 'goodbye', 'test'], 10, 0.7)
// Returns Map<string, SearchResult[]>
for (const [query, queryResults] of results) {
  console.log(`${query}: ${queryResults.length} results`)
}
```

#### `keysIterator()`

Returns an async iterator over all keys. More memory-efficient than `keys()`.

```typescript
for await (const key of engine.keysIterator()) {
  console.log(key)
}
```

#### `searchStream(query, options?)`

Returns an async iterator that yields search results one at a time.

```typescript
for await (const result of engine.searchStream('query', {
  minSimilarity: 0.7
})) {
  console.log(result.key, result.similarity)
}
```

#### `stats()`

Returns database statistics.

```typescript
const stats = await engine.stats()
console.log(stats.recordCount) // Number of live records
console.log(stats.dataFileSize) // Data file size in bytes
console.log(stats.walFileSize) // WAL file size in bytes
console.log(stats.dimension) // Embedding dimension (e.g., 384)
console.log(stats.isReadOnly) // Whether in read-only mode
```

#### `get(key)`

Retrieve a specific entry by key.

```typescript
const entry = await engine.get('doc1')
if (entry) {
  console.log(entry.key) // 'doc1'
  console.log(entry.embedding) // [0.1, 0.2, ...] (768 dimensions)
}
```

## CLI Reference

### Commands

```bash
# Store text
olingo store <key> <text> [--storePath path]

# Search for similar text
olingo search <query> [--limit 10] [--minSimilarity 0] [--storePath path]

# Get by key
olingo get <key> [--storePath path]
```

### Options

- `-s, --storePath` - Path to database file (default: `./database.raptor`)
- `-l, --limit` - Maximum results to return (default: 10)
- `-m, --minSimilarity` - Minimum similarity threshold 0-1 (default: 0)

### Examples

```bash
# Store documents
olingo store doc1 "The quick brown fox jumps over the lazy dog"
olingo store doc2 "Machine learning is a subset of artificial intelligence"
olingo store doc3 "Bun is a fast JavaScript runtime"

# Search with default settings
olingo search "artificial intelligence"

# Search with custom limit and threshold
olingo search "AI and ML" --limit 3 --minSimilarity 0.7

# Use custom database path
olingo store key1 "Some text" --storePath ./data/custom.raptor
```

## How It Works

1. **Text → Embeddings**: Olingo uses the BGE-Base-EN model to convert text into
   768-dimensional vector embeddings
2. **Storage**: Embeddings are stored in an efficient binary format (.raptor
   files)
3. **Search**: When you search, Olingo compares your query embedding against all
   stored embeddings using cosine similarity
4. **Results**: Returns the most similar results ranked by similarity score

**Embedding Model**:
[BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en-v1.5) (768
dimensions)

## Performance

- **Batch operations**: Use `storeMany()` for faster bulk inserts
- **Memory efficient**: Reads file in 64KB chunks, handles large databases
- **Fast search**: Cosine similarity comparison across all embeddings
- **Deduplication**: Latest entry automatically used for duplicate keys
- **Embedding cache**: Use `embeddingCacheSize` to cache text-to-embedding
  mappings and avoid regenerating embeddings for repeated text inputs (~1.7KB
  per cached entry)

<details>
<summary><strong>Bundler Configuration</strong></summary>

This library uses `node-llama-cpp` which includes native bindings. When
bundling, mark these packages as external:

```js
// esbuild
{
  external: ['node-llama-cpp', '@node-llama-cpp/*']
}

// Vite
{
  build: {
    rollupOptions: {
      external: ['node-llama-cpp', /^@node-llama-cpp\/.*/]
    }
  }
}

// Webpack
{
  externals: ['node-llama-cpp', /^@node-llama-cpp\/.*/]
}

// Rollup
{
  external: ['node-llama-cpp', /^@node-llama-cpp\/.*/]
}
```

</details>

#### `backup(destPath)`

Create a consistent backup of the database. Copies the data file, WAL, and HNSW
sidecar to the destination.

```typescript
await engine.backup('./backups/database.raptor')
```

#### `compact()`

Compact the database by rewriting only live records. Removes dead records from
deletes and updates, reducing file size.

```typescript
const result = await engine.compact()
console.log(result.recordsBefore) // Total records before (including dead)
console.log(result.recordsAfter) // Live records after
console.log(result.bytesBefore) // File size before
console.log(result.bytesAfter) // File size after
```

#### `verify()`

Verify the integrity of the database by scanning all records and validating
checksums.

```typescript
const result = await engine.verify()
console.log(result.totalRecords) // Total records scanned
console.log(result.validRecords) // Valid records
console.log(result.corruptRecords) // Corrupt records
console.log(result.issues) // Array of { offset, message }
```

## Events

`EmbeddingEngine` extends `EventEmitter` and emits events for database
operations:

```typescript
engine.on('store', ({ key }) => console.log(`Stored: ${key}`))
engine.on('update', ({ key }) => console.log(`Updated: ${key}`))
engine.on('delete', ({ key }) => console.log(`Deleted: ${key}`))
engine.on('search', ({ query, resultCount }) =>
  console.log(`Search: "${query}" → ${resultCount} results`)
)
```

## Custom Embedding Providers

You can use any embedding model or service by implementing the
`EmbeddingProvider` interface:

```typescript
import { EmbeddingEngine } from 'olingo'
import type { EmbeddingProvider } from 'olingo'

const myProvider: EmbeddingProvider = {
  dimension: 1536,
  async generateEmbedding(text: string): Promise<Float32Array> {
    // Call your embedding API (OpenAI, Cohere, etc.)
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ model: 'text-embedding-3-small', input: text })
    })
    const data = await response.json()
    return new Float32Array(data.data[0].embedding)
  }
}

const engine = new EmbeddingEngine({
  storePath: './database.raptor',
  embeddingProvider: myProvider
})
```

When a custom provider is set, the default BGE model is never downloaded or
loaded.

## Performance

### Persistent HNSW Index

Olingo automatically persists the HNSW search index to a `.raptor-hnsw` sidecar
file. On startup, if the sidecar exists and is not stale, the index is loaded
from disk instead of being rebuilt from all records. This significantly improves
cold start times for large databases.

The index is saved automatically when calling `dispose()`.

## Error Handling

Olingo provides custom error classes for common failure scenarios:

- **`KeyNotFoundError`** - Thrown when attempting to update a key that doesn't
  exist. Includes the `key` property.
- **`DimensionMismatchError`** - Thrown when an embedding's dimension doesn't
  match the database's expected dimension (384 for the default model). Includes
  `expectedDimension` and `actualDimension` properties.
- **`DatabaseLockedError`** - Thrown when the database is locked by another
  process.
- **`ReadOnlyError`** - Thrown when attempting to write to a read-only database.
- **`LockPermissionError`** - Thrown when the lock file cannot be created due to
  permissions.
- **`ModelInitializationError`** - Thrown when the embedding model fails to
  initialize.
- **`EmbeddingGenerationError`** - Thrown when embedding generation fails.

```typescript
import { DimensionMismatchError } from 'olingo'

try {
  await engine.store('doc1', 'some text')
} catch (error) {
  if (error instanceof DimensionMismatchError) {
    console.log(
      `Expected ${error.expectedDimension}d, got ${error.actualDimension}d`
    )
  }
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
development setup, architecture details, and guidelines.

## License

MIT © [Christoffer Artmann](mailto:artgaard@gmail.com)
