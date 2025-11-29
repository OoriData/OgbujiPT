In-Memory Vector Store Demos. Zero-setup demos using in-memory vector stores. Perfect for initial learning and exploration, quick prototyping, embedded apps and testing.

# Prerequisites

```bash
uv pip install -U .  # Install OgbujiPT and embedding model
uv pip install sentence-transformers
```

No database setup required!

# Demos

## 1. Simple Search Demo (`simple_search_demo.py`)

Basic vector search with filtering:

```bash
python simple_search_demo.py
```

Demonstrates:
- Creating an in-memory knowledge base
- Semantic search with cosine similarity
- Metadata filtering
- Top-k retrieval

**Output:**
```
Query: "What is machine learning?"
───────────────────────────────────────────────
1. Score: 0.892 | ML basics (beginner)
   Machine learning (ML) is a subset of artificial intelligence...

2. Score: 0.745 | ML basics (beginner)
   Supervised learning uses labeled training data...
```

## 2. Chat with Memory Demo (`chat_with_memory.py`)

Conversational AI with message history:

```bash
python chat_with_memory.py
```

Demonstrates:
- Message storage with vector embeddings
- Conversation history retrieval
- Context-aware responses
- Message windowing (keep last N messages)

# Comparison with PostgreSQL Demos

| Feature | In-Memory (`memory-store/`) | PostgreSQL (`pg-hybrid/`) |
|---------|---------------------------|-------------------------|
| Setup | None - pure Python | PostgreSQL + pgvector |
| Startup | Instant | ~2-5 seconds |
| Best for | Prototyping, small data | Production, large scale |
| Persistence | No (RAM only) | Yes (disk) |
| Scale | Up to ~10K docs | Millions of docs |
| Features | Full API | Full API + advanced indexing |

# Usage Patterns

## Quick Prototype

```python
from ogbujipt.store import RAMDataDB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
db = RAMDataDB(embedding_model=model, collection_name='docs')

await db.setup()
await db.insert('Your content here', metadata={'source': 'file.txt'})

async for result in db.search('query', limit=5):
    print(result.content, result.score)

await db.cleanup()
```

## Switch to PostgreSQL Later

You can easily twitch to PostgreSQL (`DataDB`/`MessageDB`) for production applications or large datasets. It would also give you persistent storage, multi-user, advanced indexing (HNSW, IVFFlat) and more, but you will have more to administer and manage. You might also prefer other backends such as Qdrant or ChromaDB, but read PG as a catch-all for the rest of this section.

The API is identical; just change the import:

```python
# Development (no setup)
from ogbujipt.store import RAMDataDB as DataDB

# Production (PostgreSQL)
# from ogbujipt.store.postgres import DataDB

# Rest of code stays the same!
db = DataDB(...)
```

## Learn More

- [Test Store README](../../test/store/README.md) - Testing with in-memory stores
- [PG Hybrid Demos](../pg-hybrid/) - PostgreSQL-based examples
- [Contributing Guide](../../CONTRIBUTING.md) - Development setup
