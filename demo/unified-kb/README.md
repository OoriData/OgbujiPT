**UnifiedKB Demos**. Demonstrations of the `UnifiedKB` API—a unified interface for managing multiple knowledge base backends with automatic result aggregation, supporting:

- **In-memory stores** (RAMDataDB, RAMMessageDB)
- **PostgreSQL + pgvector** (DataDB, MessageDB)
- **Qdrant** vector database
- **Onya** knowledge graphs
- Any backend implementing the `KBBackend` protocol

# Key Features

- **Unified interface**: One API for all backends
- **Automatic aggregation**: Results from multiple backends automatically merged and sorted
- **Parallel execution**: Searches run concurrently across backends using asyncio
- **Backend management**: Enable/disable backends without removing them
- **Selective operations**: Target all backends or specific ones
- **Weight-based scoring**: Prioritize results from certain backends
- **Error isolation**: One backend failure doesn't break the entire operation

# Demos

## `simple_unified_demo.py`

Complete introduction to UnifiedKB showing:

1. Setting up multiple backend stores
2. Registering backends with UnifiedKB
3. Inserting documents across all backends
4. Searching and aggregating results
5. Searching specific backends only
6. Managing backends (enable/disable)
7. Selective insertion to specific backends

**Requirements:**
```bash
uv pip install ogbujipt sentence-transformers
```

**Run:**
```bash
python demo/unified-kb/simple_unified_demo.py
```

**Expected output:**
```
======================================================================
UnifiedKB Demo: Unified Knowledge Base API
======================================================================

[1] Loading embedding model...
    ✓ Model loaded: all-MiniLM-L6-v2

[2] Setting up backend stores...
    ✓ Backend 1: General programming KB (in-memory)
    ✓ Backend 2: Language-specific KB (in-memory)
    ✓ Backend 3: Cache KB (in-memory)

[3] Creating UnifiedKB and registering backends...
    ✓ Registered 3 backends

[4] Backend information:
    • general:
      - Weight: 1.5
      - Enabled: True
      - Scope: general
    ...
```

# Usage Patterns

## Basic Setup

```python
from ogbujipt.memory.unified import UnifiedKB
from ogbujipt.store.ram import RAMDataDB
from sentence_transformers import SentenceTransformer

# Create backends
model = SentenceTransformer('all-MiniLM-L6-v2')
backend1 = RAMDataDB(embedding_model=model, collection_name='kb1')
await backend1.setup()

# Create unified KB
kb = UnifiedKB()
kb.add_backend('main', backend1, weight=1.5,
              metadata={'type': 'vector', 'scope': 'general'})
```

## Searching All Backends

```python
# Searches all enabled backends in parallel
async for result in kb.search('machine learning', limit=10):
    print(f'{result.score:.3f} [{result.source}]: {result.content}')
```

## Searching Specific Backends

```python
# Search only certain backends
async for result in kb.search('query', backends=['main', 'cache']):
    print(result.content)
```

## Inserting Content

```python
# Insert to all enabled backends (default)
result_ids = await kb.insert(
    'Python is great for ML',
    metadata={'topic': 'programming'}
)

# Insert to specific backends only
result_ids = await kb.insert(
    'Temporary data',
    backends=['cache'],
    metadata={'ttl': 3600}
)
```

## Managing Backends

```python
# List all backends
for name, info in kb.list_backends().items():
    print(f'{name}: weight={info["weight"]}, enabled={info["enabled"]}')

# Temporarily disable a backend
kb.disable_backend('slow_backend')

# Re-enable it later
kb.enable_backend('slow_backend')

# Remove backend completely
kb.remove_backend('old_backend')
```

## Backend Weights

Weights influence result scoring during aggregation:

```python
# Higher weight = results from this backend ranked higher
kb.add_backend('primary', pg_store, weight=2.0)    # Most important
kb.add_backend('secondary', ram_store, weight=1.0)  # Standard
kb.add_backend('cache', cache_store, weight=0.5)    # Less important
```

# Design Philosophy

## Composability over Monolith

UnifiedKB is a thin orchestration layer that doesn't duplicate backend capabilities—it coordinates them. Each backend maintains its own:
- Connection pools
- Embedding models
- Indexing strategies
- Configuration

## Explicit over Implicit

No hidden magic:
- Parallel execution is explicit (using asyncio.gather())
- Error handling is clear (one backend failure doesn't break others)
- Backend selection is explicit (via `backends` parameter)
- Scoring/aggregation is transparent (simple sort by score)

## Protocol-based Design

Any object implementing the `KBBackend` protocol works:

```python
class MyCustomBackend:
    async def search(self, query, limit=5, **kwargs):
        # Your implementation
        ...

    async def insert(self, content, metadata=None, **kwargs):
        # Your implementation
        ...

    async def delete(self, item_id, **kwargs):
        # Your implementation
        ...
```

# Advanced Topics

## Result Aggregation

Currently, UnifiedKB uses simple score-based aggregation:
1. Execute searches in parallel across all selected backends
2. Collect all results
3. Sort by score (descending)
4. Return top N results

**Coming soon:**
- Score normalization across different backend types
- Result deduplication (fuzzy matching)
- Cross-encoder re-ranking
- Query classification and intelligent routing

## Error Handling

Backend operations are isolated:
- Search: Failed backends are logged but don't block others
- Insert: Partial success is possible (some backends succeed, others fail)
- Delete: Each backend's result tracked independently

## Performance Considerations

- **Parallel execution**: All backend operations run concurrently
- **Streaming results**: Results yielded as they're aggregated (not all buffered)
- **Timeout support**: Can be added per-backend if needed
- **Connection pooling**: Handled by each backend independently

# Next Steps

After exploring this demo, check out:
- `demo/ram-store/` - In-memory backend examples
- `demo/pg-hybrid/` - PostgreSQL hybrid search examples
- `demo/kgraph/` - Onya knowledge graph examples
- `test/memory/test_unified.py` - Comprehensive unit tests

# Roadmap

Upcoming UnifiedKB features:
- Query classification for intelligent routing
- Score normalization strategies
- Result deduplication (fuzzy matching)
- Integration with search strategies (hybrid, sparse, dense)
- MCP (Model Context Protocol) support
- Advanced aggregation (RRF, cross-encoder reranking)
