# Contributing to Olingo

Thank you for your interest in contributing to Olingo! This guide will help you
get started with development.

## Getting Started

### Prerequisites

- [Bun](https://bun.sh) v1.0 or higher
- Git

### Clone and Install

```bash
git clone https://github.com/artmann/olingo.git
cd olingo
bun install
```

## Development Workflow

### Available Scripts

```bash
# Development
bun run dev          # Run the main entry point
bun run cli          # Run CLI in development mode

# Testing
bun test             # Run tests once
bun test:watch       # Run tests in watch mode
bun test:ui          # Run tests with UI

# Code Quality
bun run typecheck    # TypeScript type checking
bun run lint         # Run ESLint
bun run lint:fix     # Auto-fix linting issues
bun run format       # Format code with Prettier
bun run format:check # Check formatting

# Build
bun run build        # Build for distribution (ESM + CJS + types)
```

### Development Loop

1. Make your changes
2. Run tests: `bun test`
3. Check types: `bun run typecheck`
4. Lint code: `bun run lint`
5. Format code: `bun run format`
6. Commit your changes

## Project Architecture

### Core Components

#### 1. EmbeddingEngine (`src/engine.ts`)

The main class that orchestrates all operations:

- `store(key, text)` - Stores text with auto-generated embedding
- `storeMany(items)` - Batch store multiple entries (faster)
- `get(key)` - Retrieves entry by key (most recent if duplicates)
- `search(query, limit, minSimilarity)` - Performs semantic search
- `generateEmbedding(text)` - Generates embeddings using FastEmbed
- `cosineSimilarity(a, b)` - Calculates cosine similarity between vectors

#### 2. BinaryFileReader (`src/binary-file-reader.ts`)

Memory-efficient file reader that:

- Reads binary files in reverse order (most recent entries first)
- Processes file in 64KB chunks to handle large files
- Automatically deduplicates by key (newest entry wins)
- Uses async generator pattern for streaming

#### 3. Binary Format (`src/binary-format.ts`)

Handles reading and writing the binary storage format:

- `writeHeader()` - Writes file header with magic bytes and metadata
- `writeRecord()` - Writes single record
- `writeRecords()` - Batch writes multiple records
- `readHeader()` - Reads and validates file header
- `readRecordForward()` - Reads a single record

#### 4. CandidateSet (`src/candidate-set.ts`)

Optimized data structure for top-N search results:

- Maintains fixed-size priority queue of highest similarity matches
- Early rejection of candidates below minimum threshold
- Efficient O(n) insertion for bounded size
- Sorts results by similarity (descending)

#### 5. CLI (`src/cli.ts`)

Command-line interface with three commands:

- `olingo store <key> <text>` - Store new embedding
- `olingo get <key>` - Retrieve by key
- `olingo search <query>` - Semantic search

### Data Flow

**Store Operation:**

```
User Input → Generate Embedding → Append to Binary File → Success
```

**Search Operation:**

```
Query → Generate Embedding → Read File (Chunked) →
Calculate Similarities → CandidateSet (Top-N) → Return Results
```

**Get Operation:**

```
Key → Read File (Chunked, Reverse) → Find First Match → Return Entry
```

## Storage Format

Olingo uses a compact binary format (.raptor files):

### Header (16 bytes)

- **Magic bytes**: "EMBD" (4 bytes)
- **Version**: 1 (2 bytes, uint16, little-endian)
- **Embedding dimension**: 768 for BGE-Base-EN (4 bytes, uint32, little-endian)
- **Reserved**: 6 bytes for future use

### Records (variable length)

Each record contains:

- **Key length** (2 bytes, uint16, little-endian)
- **Key** (UTF-8 encoded, variable length)
- **Embedding vector** (dimension × 4 bytes, float32 array, little-endian)
- **Record length footer** (4 bytes, uint32, little-endian)

### Design Decisions

- **Append-only**: New entries are always appended to the end
- **Deduplication on read**: Most recent entry for a key wins
- **Reverse reading**: File is read backwards for efficient deduplication
- **No deletions**: To "delete", append a new entry with the same key
- **Single file**: All data in one .raptor file

### Benefits

- ~50% smaller than JSONL format
- Faster parsing (no JSON decode)
- Efficient reverse reading for deduplication
- Single-file storage simplicity

## Code Style

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

- **Kebab-case** for files: `embedding-file-reader.ts`
- **PascalCase** for classes: `EmbeddingEngine`, `CandidateSet`
- **camelCase** for functions and variables
- **Test files**: `*.test.ts`

### Import Organization

1. External dependencies
2. Internal modules
3. Types (using `import type`)

## Testing Guidelines

### Test Files

Each component has a corresponding `.test.ts` file:

- `engine.test.ts` - Core engine functionality
- `candidate-set.test.ts` - CandidateSet behavior
- `binary-file-reader.test.ts` - File reading logic

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

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** for any new functionality
4. **Run all checks**:
   ```bash
   bun run typecheck && bun run lint:check && bun run format:check && bun test
   ```
5. **Update documentation** if needed (README.md, CLAUDE.md)
6. **Commit with clear messages** describing what and why
7. **Push to your fork** and submit a pull request
8. **Respond to feedback** from maintainers

## CI/CD Pipeline

GitHub Actions workflow runs on:

- Push to `main` or `develop` branches
- Pull requests to `main`

**CI Jobs:**

1. **install** - Cache and install dependencies
2. **typecheck** - TypeScript type checking
3. **lint** - ESLint with zero warnings
4. **format** - Prettier formatting check
5. **test** - Run test suite
6. **build** - Build package and upload artifacts

All jobs must pass for CI to succeed.

## Embedding Model

Olingo uses [FastEmbed](https://github.com/qdrant/fastembed) with the
**BGE-Base-EN** model:

- **Model**: BAAI/bge-base-en-v1.5
- **Dimensions**: 768
- **Cache location**: `local_cache/` (auto-downloaded on first use)
- **Size**: ~200MB

To clear the model cache:

```bash
rm -rf local_cache/
```

## Performance Considerations

### Memory Efficiency

- **Chunked Reading**: Reads 64KB chunks instead of entire file
- **Streaming**: Uses async generators to avoid loading all entries
- **Early Termination**: Stops reading when enough results found

### Optimization Opportunities

- **Index Creation**: Add in-memory index for faster lookups
- **Batch Operations**: Use `storeMany()` for bulk inserts
- **Compression**: Compress binary file for storage savings
- **Caching**: Cache frequently accessed embeddings

### Scalability Limits

- Binary format requires linear scan O(n) for search
- Search performance degrades with file size
- Consider migration to indexed format for >100K entries

## Troubleshooting

### Common Issues

**"Model not found" error:**

- FastEmbed downloads model on first use to `local_cache/`
- Ensure internet connectivity on first run
- Check disk space for ~200MB model

**Slow searches:**

- Binary format scans entire file for search
- Consider reducing file size or adding indexes
- Use higher `minSimilarity` threshold to reduce candidates

**Out of memory:**

- Check file size - may need chunking optimization
- Reduce batch size in operations
- Consider splitting database into multiple files

**Type errors:**

- Run `bun run typecheck` to see all errors
- Ensure strict mode compliance
- Check `tsconfig.json` for configuration

## Need Help?

- **Issues**: [GitHub Issues](https://github.com/artmann/olingo/issues)
- **Documentation**: See [CLAUDE.md](CLAUDE.md) for detailed technical docs

## License

By contributing to Olingo, you agree that your contributions will be licensed
under the MIT License.
