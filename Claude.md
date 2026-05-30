# Claude.md - Olingo Project Guide

## Project Overview

**Olingo** is a lightweight embedding database for Node.js (>=18) and Bun (>=1)
that stores text embeddings in a WAL-based binary format (`.raptor` files). It
provides semantic search via HNSW (Hierarchical Navigable Small World) indexing
with the BGE-Small-EN-V1.5 embedding model (384 dimensions).

**Key Features:**

- Simple key-value storage for text embeddings
- Semantic search using HNSW index with cosine similarity
- WAL-based binary storage with CRC32 checksums
- Pluggable embedding providers (or bring your own embeddings)
- EventEmitter-based hooks for database operations
- Database compaction, integrity verification, and backup
- Both CLI and programmatic API
- Full TypeScript support

**Tech Stack:**

- Runtime: Node.js >=18, Bun >=1
- Language: TypeScript
- Embedding Model: node-llama-cpp (BAAI/bge-small-en-v1.5-gguf, 384 dimensions)
- Build Tool: Rolldown
- Test Framework: Vitest

## Architecture

### Core Components

#### 1. EmbeddingEngine (`src/engine.ts`)

The main class (extends `EventEmitter`) that orchestrates all operations:

- `store(key, text)` / `storeMany(items)` - Store text with auto-generated
  embedding
- `storeEmbedding(key, embedding)` / `storeManyEmbeddings(items)` - Store
  pre-computed embeddings
- `update(key, text)` - Update existing key (throws `KeyNotFoundError` if
  missing)
- `get(key)` / `has(key)` / `keys()` / `count()` - Lookup operations
- `search(query, options)` / `searchMany(queries)` - Semantic search
- `similarTo(key, options)` - Find documents most similar to an existing key
  (uses the key's stored embedding; excludes the key itself)
- `keysIterator()` / `searchStream()` - Async iterators for streaming
- `delete(key)` - Logical delete
- `stats()` / `verify()` / `compact()` / `backup()` - Database management
- `dispose()` - Cleanup resources and persist HNSW index
- Emits events: `store`, `update`, `delete`, `search`, `similarTo`

#### 2. StorageEngine (`src/storage-engine/storage-engine.ts`)

WAL-based durable storage orchestrator:

- Write path: Lock → Data write → fsync → WAL write → fsync → Index update
- Operation-level locking (acquires lock per write, releases immediately)
- Recovery via WAL replay on startup
- Compaction support (rewrite live records, rebuild WAL)

#### 3. HnswIndex (`src/storage-engine/hnsw-index.ts`)

HNSW approximate nearest-neighbor search:

- Parameters: maxLayer=3, maxNeighbors=10, efConstruction=200, efSearch=50
- Lazily built from storage on first search call
- Persistent serialization to `.raptor-hnsw` sidecar file

#### 4. KeyIndex (`src/storage-engine/key-index.ts`)

In-memory map for O(1) key lookups, rebuilt from WAL during startup.

#### 5. WAL (`src/storage-engine/wal.ts`)

Fixed 48-byte entries per operation with checksums.

#### 6. FileLock (`src/storage-engine/file-lock.ts`)

Atomic exclusive file-based locking with stale lock detection via PID checking.

#### 7. CLI (`src/cli.ts`)

Command-line interface with commands: store, get, search, similar-to, delete,
count, keys, stats, verify, compact, wal.

### Data Flow

**Store Operation:**

```
User Input → Generate Embedding → Lock → Write Data → fsync →
Write WAL → fsync → Update Index → Unlock → Emit 'store'
```

**Search Operation:**

```
Query → Generate Embedding → Ensure HNSW Index →
HNSW Search → Filter by minSimilarity → Sort → Emit 'search'
```

**Get Operation:**

```
Key → KeyIndex Lookup (O(1)) → Read Record at Offset → Return Entry
```

## File Structure

```
src/
├── commands/                  # CLI command implementations
│   ├── flags.ts              # Shared CLI flags (--storePath)
│   ├── compact.ts            # Compact command
│   ├── count.ts              # Count command
│   ├── delete.ts             # Delete command
│   ├── get.ts                # Get command
│   ├── keys.ts               # Keys command
│   ├── search.ts             # Search command
│   ├── stats.ts              # Stats command
│   ├── store.ts              # Store command
│   ├── verify.ts             # Verify command
│   ├── wal.ts                # WAL display command
│   └── index.ts              # Command exports
├── storage-engine/            # Storage layer
│   ├── storage-engine.ts     # WAL-based storage orchestrator
│   ├── hnsw-index.ts         # HNSW approximate nearest neighbor search
│   ├── hnsw-persistence.ts   # HNSW index serialization/deserialization
│   ├── key-index.ts          # In-memory key → location map
│   ├── wal.ts                # Write-ahead log
│   ├── data-format.ts        # Binary serialization with CRC32
│   ├── file-lock.ts          # File-based locking with stale detection
│   ├── integrity.ts          # Database integrity verification
│   ├── dimension-mismatch-error.ts  # Dimension validation error
│   ├── constants.ts          # Magic numbers, sizes
│   ├── types.ts              # Storage-layer types
│   └── index.ts              # Storage exports
├── candidate-set.ts          # Top-N search results data structure
├── cli.ts                    # CLI entry point
├── engine.ts                 # Core EmbeddingEngine class (extends EventEmitter)
├── key-not-found-error.ts    # KeyNotFoundError class
├── lru-cache.ts              # Generic LRU cache
├── index.ts                  # Library exports
└── types.ts                  # TypeScript definitions
```

## Development Workflows

### Setup

```bash
bun install
```

### Development

```bash
bun run dev          # Run main entry point
bun run cli          # Run CLI in development mode
```

### Code Quality

```bash
bun run typecheck    # TypeScript type checking
bun run lint         # Run ESLint
bun run lint:fix     # Auto-fix linting issues
bun run format       # Format code with Prettier
bun run format:check # Check formatting
```

### Testing

```bash
bun run test         # Run tests once
bun run test:watch   # Run tests in watch mode
bun run test:ui      # Run tests with UI
```

### Build

```bash
bun run build        # Build for distribution (ESM + CJS + types)
```

## Common Development Tasks

### Adding a New Command

1. Create command file in `src/commands/`
2. Import shared flags from `src/commands/flags.ts`
3. Use `EmbeddingEngine` for operations
4. Export from `src/commands/index.ts`
5. Register in `src/cli.ts`

### Modifying Storage Format

- Update types in `src/storage-engine/types.ts`
- Modify `serializeDataRecord` / `deserializeDataRecord` in
  `src/storage-engine/data-format.ts`
- Update `StorageEngine` methods in `src/storage-engine/storage-engine.ts`
- Add migration logic if needed (see `ensureV2Format`)

### Changing Embedding Model

- Update node-llama-cpp model initialization in `src/engine.ts`
- Update `defaultDimension` constant (currently 384 for BGE-Small-EN-V1.5)
- Clear `.cache/models/` to download new model
- Update README with new model details
- Or: use the `embeddingProvider` option in `EngineOptions` for custom models

### Adding New Search Filters

- Extend `SearchOptions` in `src/types.ts`
- Modify `search()` method in `src/engine.ts`
- Add CLI flags in `src/commands/search.ts`

## Code Style and Conventions

### TypeScript Rules

- **Strict Mode**: All strict TypeScript checks enabled
- **No `any`**: Use proper types or `unknown`
- **No Non-null Assertions**: Avoid `!` operator
- **Prefer Nullish Coalescing**: Use `??` over `||`
- **No Floating Promises**: Always await or handle promises
- **No Unused Vars**: Prefix with `_` if intentionally unused

### Formatting (Prettier)

- Single quotes
- No semicolons
- 2-space indentation
- Trailing commas: none
- Arrow function parens: always

### File Naming

- Kebab-case for files: `embedding-file-reader.ts`
- Test files: `*.test.ts`
- PascalCase for classes: `EmbeddingEngine`, `CandidateSet`
- camelCase for functions and variables

### Import Organization

1. External dependencies
2. Internal modules
3. Types (using `import type`)

## Testing Guidelines

### Test Files

Each component has a corresponding `.test.ts` file:

- `engine.test.ts` - Core engine functionality
- `candidate-set.test.ts` - CandidateSet behavior
- `lru-cache.test.ts` - LRU cache behavior
- `embedding-provider.test.ts` - Custom provider integration
- `byoe.test.ts` - Bring-your-own-embeddings
- `update.test.ts` - Update method
- `events.test.ts` - EventEmitter events
- `search-enrichment.test.ts` - DetailedSearchResult
- `streaming.test.ts` - Async iterators
- `batch-search.test.ts` - Batch search
- `stats.test.ts` - Database statistics
- `backup.test.ts` - Backup & restore
- `storage-engine/hnsw-index.test.ts` - HNSW index
- `storage-engine/hnsw-persistence.test.ts` - HNSW serialization
- `storage-engine/integrity.test.ts` - Integrity verification
- `storage-engine/compaction.test.ts` - Compaction
- `storage-engine/dimension-mismatch-error.test.ts` - Dimension validation
- `storage-engine/stale-lock.test.ts` - Stale lock detection
- `storage-engine/integration/*.test.ts` - Integration tests (baseline,
  concurrency, corruption, crash, locking, recovery, etc.)

### Testing Patterns

- Use `describe()` blocks to group related tests
- Use `it()` for individual test cases
- Clean up test data in `afterEach()` or `afterAll()`
- Use `expect()` assertions from Vitest
- Mock file system operations when needed
- Test edge cases: empty files, duplicates, large datasets

### Running Specific Tests

```bash
bun test src/engine.test.ts          # Run single file
bun test -t "search returns results" # Run by pattern
```

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on:

- Push to `main` or `develop` branches
- Pull requests to `main`

**Jobs:**

1. **install** - Cache and install dependencies
2. **typecheck** - TypeScript type checking
3. **lint** - ESLint with zero warnings (`lint:check`)
4. **format** - Prettier formatting check
5. **test** - Run test suite
6. **build** - Build package and upload artifacts

All jobs must pass for CI to succeed.

### Release Pipeline

A separate release workflow (`.github/workflows/release.yml`) triggers on
version tags (`v*`):

1. Runs all quality checks (typecheck, lint, format, test)
2. Builds the package
3. Publishes to npm with provenance

To release: `git tag v3.0.0 && git push --tags`

## Storage Format

### Binary Format (`.raptor`)

Records are stored in a binary format with CRC32 checksums:

```
Record: magic(4) + version(2) + opType(1) + flags(1) + seqNum(8) +
        timestamp(8) + keyLen(2) + key(N) + dimension(4) +
        embedding(D*4) + checksum(4) + trailer(4)
```

WAL entries are fixed 48-byte records for durability.

### Important Notes

- **WAL-based durability**: Write-ahead log ensures crash safety
- **Deduplication**: Most recent entry for a key wins (by sequence number)
- **Default Path**: `./database.raptor`
- **Embedding Dimensions**: 384 (BGE-Small-EN-V1.5 model)
- **Dimension Validation**: `DimensionMismatchError` thrown on mismatch
- **Logical Deletion**: Delete markers recorded, compaction reclaims space
- **Compaction**: `compact()` rewrites only live records

## API Reference

### Programmatic Usage

```typescript
import { EmbeddingEngine } from 'olingo'

const engine = new EmbeddingEngine({
  storePath: './my-database.raptor'
})

// Store
await engine.store('doc1', 'Machine learning is awesome')
await engine.storeMany([{ key: 'doc2', text: 'Deep learning' }])

// Search
const results = await engine.search('AI and ML', {
  limit: 5,
  minSimilarity: 0.7
})
const batchResults = await engine.searchMany(['AI', 'ML'], 5, 0.7)

// CRUD
const entry = await engine.get('doc1')
await engine.update('doc1', 'Updated text')
await engine.delete('doc1')

// Management
const stats = await engine.stats()
await engine.compact()
const verifyResult = await engine.verify()
await engine.backup('./backup.raptor')

// Always dispose when done
await engine.dispose()
```

### CLI Usage

```bash
# Data operations
olingo store doc1 "Machine learning is awesome"
olingo search "AI and ML" --limit 5 --minSimilarity 0.7
olingo similar-to doc1 --limit 5
olingo get doc1
olingo delete doc1

# Database management
olingo count
olingo keys
olingo stats
olingo verify
olingo compact

# Custom database path
olingo store doc1 "text" --storePath ./custom.raptor
```

## Performance Considerations

### Search Performance

- **HNSW Index**: O(log N) approximate nearest neighbor search
- **Persistent HNSW**: Index saved to `.raptor-hnsw` sidecar, loaded on startup
- **Key Lookup**: O(1) via in-memory KeyIndex

### Write Performance

- **Operation-level locking**: Locks acquired per write, not held across session
- **WAL-based durability**: fsync after each write for crash safety
- **Batch operations**: `storeMany()` for bulk inserts

### Memory

- **HNSW index**: Loaded entirely in memory
- **LRU cache**: Optional embedding cache (`embeddingCacheSize`)
- **Compaction**: `compact()` reclaims space from deleted/updated records

## Troubleshooting

### Common Issues

**"Model not found" error:**

- node-llama-cpp downloads BGE-Small-EN-V1.5 model on first use to
  `.cache/models/`
- Ensure internet connectivity on first run
- Check disk space for ~67MB model

**Stale lock file:**

- Olingo auto-detects stale locks via PID checking
- If a process crashes, the lock is recovered automatically on next access

**Slow searches:**

- HNSW index is built lazily on first search (can be slow for large databases)
- Subsequent searches use the persistent HNSW index from disk
- Use `compact()` to remove dead records and reduce index size

**Type errors:**

- Run `bun run typecheck` to see all errors
- Ensure strict mode compliance
- Check `tsconfig.json` for configuration

## Platform-Specific Notes

### Bun on Windows

**File open with numeric constants bug:**

Bun on Windows has a bug where `fs.open()` fails with `ENOENT` when using
numeric constants (`constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY`),
even when the parent directory exists. The workaround is to use string flags
instead:

```typescript
// Broken on Bun/Windows:
import { constants } from 'node:fs'
await open(path, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY)

// Works everywhere:
await open(path, 'wx') // 'wx' = O_CREAT | O_EXCL | O_WRONLY
```

This affects the `FileLock` class in `src/storage-engine/file-lock.ts`.

## Additional Resources

- **CODE_STYLE.md** - Detailed code style guide and conventions
- **README.md** - User-facing documentation
- **package.json** - Scripts and dependencies
- **tsconfig.json** - TypeScript configuration
- **.eslintrc.json** - Linting rules
- **rolldown.config.ts** - Build configuration
- **vitest.config.ts** - Test configuration

## Quick Commands Reference

| Task                 | Command                                                                       |
| -------------------- | ----------------------------------------------------------------------------- |
| Install dependencies | `bun install`                                                                 |
| Run tests            | `bun test`                                                                    |
| Type check           | `bun run typecheck`                                                           |
| Lint code            | `bun run lint`                                                                |
| Format code          | `bun run format`                                                              |
| Build package        | `bun run build`                                                               |
| Run CLI              | `bun run cli`                                                                 |
| Run all checks       | `bun run typecheck && bun run lint:check && bun run format:check && bun test` |
