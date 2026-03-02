# Olingo Examples

This directory contains practical examples demonstrating different use cases for
the Olingo embedding database.

## Running Examples

Each example is self-contained and can be run directly with Bun:

```bash
bun run examples/01-document-search.ts
bun run examples/02-faq-bot.ts
bun run examples/03-code-snippets.ts
bun run examples/04-content-recommendation.ts
```

## Examples

### 1. Document Search / Simple RAG

**File:** `01-document-search.ts`

Demonstrates semantic search over documentation chunks - a basic RAG (Retrieval
Augmented Generation) pattern. Store multiple documentation sections and search
with natural language queries.

**Use cases:**

- Searchable knowledge bases
- Internal documentation search
- Context retrieval for LLM prompts

### 2. FAQ Bot

**File:** `02-faq-bot.ts`

Shows how to build a FAQ matching system that finds the most relevant
question-answer pairs based on user queries.

**Use cases:**

- Customer support chatbots
- Help center search
- Automated question answering

### 3. Code Snippet Library

**File:** `03-code-snippets.ts`

Search code snippets by natural language description. Store code examples with
descriptions and find them using what they do rather than exact keywords.

**Use cases:**

- Developer documentation
- Code example libraries
- Internal snippet repositories

### 4. Content Recommendation

**File:** `04-content-recommendation.ts`

Implements "more like this" functionality by finding semantically similar
content items.

**Use cases:**

- Blog post recommendations
- Product suggestions
- Related content discovery

## Notes

- Each example uses a temporary database file that's cleaned up after running
- All examples include sample data inline for easy experimentation
- Modify the queries and data to test with your own content
