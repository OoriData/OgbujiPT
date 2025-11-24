This PostgreSQL Hybrid Search Demo demonstrates OgbujiPT's Sparse Retrieval (BM25) and Hybrid Search (RRF) capabilities. The combination of dense vector search with sparse keyword-based retrieval should yield superior results.

# What's Hybrid Search?

**Hybrid search** combines multiple retrieval strategies to leverage their complementary strengths:

- **Dense vectors** (embeddings): Capture semantic meaning, great for conceptual similarity
- **Sparse vectors** (BM25): Capture exact keyword matches, great for terminology and names

**Reciprocal Rank Fusion (RRF)** merges results from both approaches, consistently outperforming either method alone.

# Demos in this Directory

## 1. `hybrid_search.ipynb`

For learning the fundamentals interactively, this notebook covers:
- Setting up hybrid search with BM25 + dense vectors
- Comparing dense-only vs hybrid retrieval
- Tuning BM25 parameters (k1, b)
- Understanding RRF fusion

## 2. `chat_with_hybrid_kb.py`
For cut & paste ready, application patterns, an advanced conversational AI demo with:
- Chat history tracking (MessageDB)
- Knowledge base with hybrid search (DataDB + BM25)
- Context-aware responses using both conversation and KB

# Prerequisites

## 1. PostgreSQL with pgvector Extension

### Docker install

Easiest way is using Docker. We recommend the official pgvector image:

```bash
docker run --name pg_hybrid_demo \
  -p 5432:5432 \
  -e POSTGRES_USER=demo_user \
  -e POSTGRES_PASSWORD=demo_pass_2025 \
  -e POSTGRES_DB=hybrid_demo \
  -d pgvector/pgvector
```

<!-- 
```sh
docker pull pgvector/pgvector:pg17
docker run --name mock-postgres -p 5432:5432 \
    -e POSTGRES_USER=demo_user -e POSTGRES_PASSWORD=demo_pass_2025 -e POSTGRES_DB=mock_db \
    -d pgvector/pgvector:pg17
```
-->

This provides:
- PostgreSQL with pgvector 0.7.0+ (required for sparse vectors)
- Running on `localhost:5432`
- Database: `hybrid_demo`
- User: `demo_user` / Password: `demo_pass_2025`

### Existing DB

You can use an existing PostgreSQL instance with pgvector extension. OgbujiPT should install it for you, but configs can differ, so you might have to do your own equivalent of:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

You can then use e.g. a password manager for environment injection. For example, with 1passwd commands like:

```sh
op run --no-masking --env-file=.env -- python chat_with_hybrid_kb.py
```

Where you have a `.env` file with your the `PG_DB_` variables, including secret references where suitable. Further reading: (https://huggingface.co/blog/ucheog/separate-env-setup-from-code).

## 2. Python Dependencies

Install OgbujiPT with dev dependencies:

```bash
# From the OgbujiPT root directory
uv pip install -U .

# Additional demo dependencies
uv pip install jupyter sentence-transformers
```

This will automatically install other core dependencies, such as:
- `rank-bm25`: BM25 sparse retrieval
- `pgvector`: PostgreSQL vector operations
- `asyncpg`: Async PostgreSQL driver
- `structlog`: Structured logging

# Quick Start

## Option A: Jupyter Notebook (Interactive)

```bash
cd demo/pg-hybrid
jupyter notebook hybrid_search.ipynb
```

Follow the cells to:
1. Connect to PostgreSQL
2. Index sample documents
3. Compare dense vs sparse vs hybrid search
4. Experiment with parameters

## Option B: Python Script (Full Application)

```bash
cd demo/pg-hybrid
python chat_with_hybrid_kb.py
```

This runs an interactive chat session that:
- Maintains conversation history
- Searches a knowledge base with hybrid retrieval
- Provides context-aware responses

# Configuration

Both demos use these default connection parameters:

```python
DB_HOST = 'localhost'
DB_PORT = 5432
DB_NAME = 'hybrid_demo'
DB_USER = 'demo_user'
DB_PASSWORD = 'demo_pass_2025'
```

To use different settings, edit the connection parameters in the demo files or set environment variables:

```bash
export PG_HOST=my-postgres-host
export PG_PORT=5432
export PG_DB=my_database
export PG_USER=my_user
export PG_PASSWORD=my_password
```

# Understanding the Code

## BM25 Sparse Retrieval

```python
from ogbujipt.retrieval import BM25Search

bm25 = BM25Search(
    k1=1.5,      # Term frequency saturation (1.2-2.0 typical)
    b=0.75,      # Document length normalization (0-1)
    epsilon=0.25 # IDF floor
)

results = bm25.execute(
    query='machine learning algorithms',
    backends=[knowledge_db],
    limit=5
)
```

Sparse retrieval is effective for queries with specific terminology, names, or exact phrases.

### Hybrid Search (Dense + Sparse)

```python
from ogbujipt.retrieval import HybridSearch, BM25Search

# Assume you have a dense search strategy already
hybrid = HybridSearch(
    strategies=[dense_search, BM25Search()],
    k=60  # RRF constant (smaller = more weight to top results)
)

results = hybrid.execute(
    query='machine learning algorithms',
    backends=[knowledge_db],
    limit=10
)
```

Hybrid Search is best for general-purpose search; it's the best of both worlds.

## Sparse Vector Storage (Advanced)

For storing sparse vectors directly (e.g., precomputed BM25 vectors):

```python
from ogbujipt.store.postgres import SparseDB

sparse_db = await SparseDB.from_conn_params(
    embedding_model=bm25_encoder,  # Your sparse encoder
    table_name='sparse_vectors',
    vocab_size=10000,  # Vocabulary dimension
    host=DB_HOST,
    port=DB_PORT,
    db_name=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    itypes=['sparsevec'],
    ifuncs=['cosine']
)

# Insert with sparse vector
await sparse_db.insert(
    content='document text',
    sparse_vector={42: 0.5, 100: 0.8, 333: 0.3}  # Only non-zero indices
)
```

# Architecture Notes

## Search Strategy Protocol

All search strategies implement the `SearchStrategy` protocol:

```python
class SearchStrategy(Protocol):
    async def execute(
        self,
        query: str,
        backends: list[KBBackend],
        limit: int = 5,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        ...
```

This allows composing strategies flexibly (e.g., dense + sparse + graph traversal).

## Result Format

All search methods yield `SearchResult` dataclass:

```python
@dataclass
class SearchResult:
    content: str           # Document text
    score: float          # Normalized 0-1 (higher = more relevant)
    metadata: dict        # Custom metadata
    source: str           # Which backend/strategy produced this
```

Scores are normalized to [0, 1] across all strategies for fair comparison.

# Troubleshooting

## Connection Refused

```
OSError: [Errno 61] Connect call failed ('127.0.0.1', 5432)
```

**Solution**: Ensure PostgreSQL is running:
```bash
docker ps  # Check if pg_hybrid_demo container is running
docker start pg_hybrid_demo  # Start if stopped
```

## Extension Not Found

```
ERROR: type "vector" does not exist
```

**Solution**: The pgvector extension isn't enabled. If using `ankane/pgvector` Docker image, it should be automatic. Otherwise:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Import Errors

```
ModuleNotFoundError: No module named 'ogbujipt.retrieval'
```

**Solution**: Install OgbujiPT:
```bash
uv pip install -U .  # From OgbujiPT root directory
```

## BM25 Returns No Results

**Likely cause**: Documents weren't indexed. BM25Search needs to build an index on first use.

**Solution**: Ensure you call `execute()` with backends that have documents, and the index will be built automatically.

# Performance Tips

1. **BM25 indexing**: For large corpora (>100k docs), BM25Search loads all documents into memory. Consider:
   - Using backends with built-in BM25 support
   - Implementing incremental indexing
   - Filtering backends before indexing

2. **Hybrid fusion**: Fetch 2-3x more results per strategy than your final limit:
   ```python
   hybrid = HybridSearch(
       strategies=[dense, sparse],
       strategy_limits={'DenseSearch': 20, 'BM25Search': 20}  # Fetch 20 each
   )
   results = hybrid.execute(query, backends, limit=10)  # Return top 10 after fusion
   ```

3. **PostgreSQL**: Use HNSW indices for large vector tables:
   ```python
   db = await DataDB.from_conn_params(
       ...,
       itypes=['vector'],    # Build HNSW index
       ifuncs=['cosine'],
       i_max_conn=16,        # Increase for better recall
       ef_construction=64    # Increase for better quality
   )
   ```

# Next Steps

- Read the [Phase 2 Architecture Notes](../../ARCHITECTURE.md) (if available)
- Explore combining with graph RAG using [Onya](https://github.com/OoriData/Onya)
- Implement reranking with cross-encoders
- Add query expansion or pseudo-relevance feedback

# References

- **BM25**: Robertson, S. E., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*
- **RRF**: Cormack, G. V., et al. (2009). *Reciprocal rank fusion outperforms condorcet and individual rank learning methods*
- **pgvector**: [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
- **Hybrid Search**: [Weaviate's Hybrid Search Explained](https://weaviate.io/blog/hybrid-search-explained)

---

**Need help?** Open an issue at [OgbujiPT GitHub](https://github.com/OoriData/OgbujiPT/issues)
