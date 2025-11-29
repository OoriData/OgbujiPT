![ogbujipt github header](https://github.com/OoriData/OgbujiPT/assets/43561307/1a88b411-1ce2-43df-83f0-c9c39d6679bc)


**OgbujiPT** is a general-purpose knowledge bank system for LLM-based applications. It provides a unified API for storing, retrieving, and managing semantic knowledge across multiple backends, with support for dense vector search, sparse retrieval, hybrid search, and more.

Built with Pythonic simplicity and transparency in mind; avoiding the over-frameworks that plague the LLM ecosystem. Every abstraction must justify its existence.

<table><tr>
  <td><a href="https://oori.dev/"><img src="https://www.oori.dev/assets/branding/oori_Logo_FullColor.png" width="64" /></a></td>
  <td>OgbujiPT is primarily developed by the crew at <a href="https://oori.dev/">Oori Data</a>. We offer data pipelines and software engineering services around AI/LLM applications.</td>
</tr></table>

[![PyPI - Version](https://img.shields.io/pypi/v/ogbujipt.svg)](https://pypi.org/project/ogbujipt)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ogbujipt.svg)](https://pypi.org/project/ogbujipt)

## Quick links

- [Getting started](#getting-started)
- [Knowledge bank features](#knowledge-bank-features)
- [LLM integration](#llm-integration)
- [License](#license)

-----

## Getting started

```console
uv pip install ogbujipt
```

### Quick example: In-memory knowledge bank

Perfect for prototyping, testing, or small applications—no database setup required:

```py
import asyncio
from ogbujipt.store import RAMDataDB
from sentence_transformers import SentenceTransformer

async def main():
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create in-memory knowledge base
    kb = RAMDataDB(embedding_model=model, collection_name='docs')
    await kb.setup()
    
    # Insert documents
    await kb.insert('Python is great for machine learning', metadata={'lang': 'python'})
    await kb.insert('JavaScript powers modern web applications', metadata={'lang': 'js'})
    
    # Semantic search
    async for result in kb.search('programming languages', limit=5):
        print(f'{result.content} (score: {result.score:.3f})')
    
    await kb.cleanup()

asyncio.run(main())
```

### Hybrid search with reranking

Combine dense vector search with sparse BM25 retrieval, then rerank for best results:

```py
from ogbujipt.retrieval.hybrid import RerankedHybridSearch
from ogbujipt.retrieval.sparse import BM25Search
from ogbujipt.retrieval.dense import DenseSearch
from rerankers import Reranker

# Initialize components
reranker = Reranker(model_name='BAAI/bge-reranker-base')
hybrid = RerankedHybridSearch(
    strategies=[DenseSearch(), BM25Search()],
    reranker=reranker,
    rerank_top_k=20
)

# Search across knowledge bases
async for result in hybrid.execute('machine learning', backends=[kb], limit=5):
    print(f'{result.score:.3f}: {result.content[:50]}...')
```

## Knowledge bank features

OgbujiPT provides a flexible knowledge bank system with multiple storage backends and retrieval strategies.

### Storage backends

- **In-memory (`RAMDataDB`, `RAMMessageDB`)**: Zero-setup stores perfect for testing, prototyping, and small applications. Drop-in replacements for PostgreSQL versions with identical APIs.
- **PostgreSQL + pgvector**: Production-ready persistent storage with advanced indexing (HNSW, IVFFlat) and full SQL capabilities.
- **Qdrant**: High-performance vector database with distributed capabilities.

### Retrieval strategies

- **Dense vector search**: Semantic similarity using embeddings (e.g., SentenceTransformers, OpenAI embeddings)
- **Sparse retrieval**: BM25 keyword-based search for exact term matching
- **Hybrid search**: Combine multiple strategies using Reciprocal Rank Fusion (RRF)
- **Reranking**: Cross-encoder reranking for improved precision (e.g., BGE-reranker, ZeRank)

### Message/conversation storage

Store and search chat history with semantic retrieval:

```py
from ogbujipt.store import RAMMessageDB
from uuid import uuid4
from datetime import datetime, timezone

async def chat_example():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    db = RAMMessageDB(embedding_model=model, collection_name='chat')
    await db.setup()
    
    conversation_id = uuid4()
    
    # Store messages
    await db.insert(conversation_id, 'user', 'What is machine learning?',
                   datetime.now(tz=timezone.utc), {})
    await db.insert(conversation_id, 'assistant', 'ML is a subset of AI...',
                   datetime.now(tz=timezone.utc), {})
    
    # Semantic search over conversation
    results = await db.search(conversation_id, 'AI concepts', limit=2)
    for msg in results:
        print(f'[{msg.role}] {msg.content}')
    
    await db.cleanup()
```

### Design philosophy

- **Composability over monoliths**: Mix and match backends and strategies
- **Explicit over implicit**: No hidden magic—you control connection pooling, retries, caching
- **Pythonic simplicity**: Minimal abstractions, clear APIs, sensible defaults
- **Production-ready**: Structured logging, retry logic, async-first design

## LLM integration

OgbujiPT includes LLM wrapper utilities for integrating knowledge banks with language models.

### Basic LLM usage

```py
from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat

llm_api = openai_chat_api(base_url='http://localhost:8000')
prompt = 'Write a short birthday greeting for my star employee'

resp = llm_api.call(prompt_to_chat(prompt), temperature=0.1, max_tokens=256)
print(resp.first_choice_text)
```

### Asynchronous API

```py
import asyncio
from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat

llm_api = openai_chat_api(base_url='http://localhost:8000')
messages = prompt_to_chat('Hello!', system='You are a helpful AI agent…')
resp = await llm_api(messages, temperature=0.1, max_tokens=256)
print(resp.first_choice_text)
```

### Supported LLM backends

You can use the OpenAI cloud LLM API and APIs which conform to this, including Anthropic's, local LM Studio, Ollama, etc. Users on Mac might want to check out our sister project [Toolio](https://github.com/OoriData/Toolio) which provides a local LLM inference server on Apple Silicon.

### RAG example: Chat with your documents

```py
from ogbujipt.store import RAMDataDB
from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat

# Setup knowledge base
kb = RAMDataDB(embedding_model=model, collection_name='docs')
await kb.setup()
await kb.insert('Your document content here...', metadata={'source': 'doc.pdf'})

# Retrieve relevant context
contexts = []
async for result in kb.search('user question', limit=3):
    contexts.append(result.content)

# Build RAG prompt
context_text = '\n\n'.join(contexts)
prompt = f"""Based on the following context, answer the question.

Context:
{context_text}

Question: user question"""

# Get LLM response
llm_api = openai_chat_api(base_url='http://localhost:8000')
resp = await llm_api(prompt_to_chat(prompt))
print(resp.first_choice_text)
```

## Demos and examples

See the [`demo/`](https://github.com/OoriData/OgbujiPT/tree/main/demo) directory for complete examples:

### Knowledge bank demos

- **`ram-store/`**: In-memory vector stores—zero setup, perfect for learning
  - `simple_search_demo.py`: Basic semantic search with filtering
  - `chat_with_memory.py`: Conversational AI with message history
- **`pg-hybrid/`**: PostgreSQL-based production examples
  - `chat_with_hybrid_kb.py`: Hybrid search with RRF fusion
  - `hybrid_rerank_demo.py`: Reranking with cross-encoders
  - `chat_doc_folder_pg.py`: RAG chat application

### LLM demos

- Basic LLM text completion and format correction
- Multiple simultaneous queries via multiprocessing
- OpenAI-style function calling
- Discord bot integration
- Streamlit UI for PDF chat

## Roadmap

OgbujiPT is evolving into a comprehensive knowledge bank system. Current focus (v0.10.0+):

### ✅ Implemented

- In-memory vector stores (RAMDataDB, RAMMessageDB)
- Dense vector search (PostgreSQL, Qdrant, in-memory)
- Sparse retrieval (BM25)
- Hybrid search with RRF fusion
- Cross-encoder reranking
- Message/conversation storage
- Metadata filtering

### 🚧 In progress

- GraphRAG support using [Onya](https://github.com/OoriData/Onya)
- Unified knowledge base API
- Query classification and routing
- Multi-backend aggregation

### 📋 Planned

- RSS feed ingestion and caching
- Link management with update mechanisms
- Graph curation strategies
- KB maintenance and pruning (summarization, obsolescence marking)
- RBAC and multi-tenancy
- Observability (query logging, tracing, performance monitoring)
- MCP (Model Context Protocol) provider/server
- Query sampling for refinement
- Additional backends (filesystem, Marqo, etc.)
- Multi-modal support

See [discussion #92](https://github.com/OoriData/OgbujiPT/discussions/92) for detailed roadmap and design philosophy.

## Installation

```console
uv pip install ogbujipt
```

### Optional dependencies

For specific features:

```console
# PostgreSQL + pgvector support
uv pip install "ogbujipt[postgres]"

# Qdrant support
uv pip install "ogbujipt[qdrant]"

# Reranking support
uv pip install "rerankers[transformers]"

# GraphRAG support (when available)
uv pip install "ogbujipt[graph]"
```

## Development and Contribution

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [contributor notes](https://github.com/OoriData/OgbujiPT/wiki/Notes-for-contributors) for development setup and guidelines.
Toolio, 
## Design principles

### Avoid over-frameworks

OgbujiPT deliberately avoids becoming another LangChain. We emphasize:

- **Minimal abstractions**: Every layer must justify its existence
- **Explicit over implicit**: No hidden magic—be clear about connection pooling, retries, caching
- **Configuration clarity**: Help automate config without creating configuration hell
- **Composability**: Mix and match components rather than monolithic frameworks
- **Pythonic**: Old-school Python simplicity and clarity

### Memory taxonomy

Different memory types need different strategies:

- **Conversational memory**: Recent chat history (working memory)
- **Semantic memory**: Long-term knowledge (documents, facts)
- **Scratchpad**: Temporary computation state
- **Observability logs**: Query/retrieval tracing

OgbujiPT provides explicit APIs for each, avoiding one-size-fits-all "universal memory" patterns.

## Resources

- [Against mixing environment setup with code](https://huggingface.co/blog/ucheog/separate-env-setup-from-code)
- [Quick setup for llama-cpp-python](https://github.com/uogbuji/OgbujiPT/wiki/Quick-setup-for-llama-cpp-python-backend)
- [Quick setup for Ooba](https://github.com/uogbuji/OgbujiPT/wiki/Quick-setup-for-text-generation-webui-(Ooba)-backend)

## License

Apache 2.0. For tha culture!

## Credits

Some initial ideas & code were borrowed from these projects, but with heavy refactoring:

* [ChobPT/oobaboogas-webui-langchain_agent](https://github.com/ChobPT/oobaboogas-webui-langchain_agent)
* [wafflecomposite/langchain-ask-pdf-local](https://github.com/wafflecomposite/langchain-ask-pdf-local)

## Related projects

* [mlx-tuning-fork](https://github.com/chimezie/mlx-tuning-fork)—"very basic framework for parameterized Large Language Model (Q)LoRa fine-tuning with MLX. It uses mlx, mlx_lm, and OgbujiPT, and is based primarily on the excellent mlx-example libraries but adds very minimal architecture for systematic running of easily parameterized fine tunes, hyperparameter sweeping, declarative prompt construction, an equivalent of HF's train on completions, and other capabilities."
* [living-bookmarks](https://github.com/uogbuji/living-bookmarks)—"Uses [OgbujiPT] to Help a user manage their bookmarks in context of various chat, etc."

## FAQ

### What's unique about OgbujiPT?

Unlike frameworks that try to do everything, OgbujiPT focuses on:

- **Knowledge bank primitives**: Clean APIs for storage and retrieval
- **Composability**: Mix backends and strategies without lock-in
- **Pythonic simplicity**: Minimal abstractions, clear code
- **Production-ready**: Async-first, structured logging, retry logic
- **Explicit design**: No hidden magic—you control the details

### Why not just use LangChain?

LangChain is great for many use cases, but it's also:
- Overly abstracted (hard to understand what's happening)
- Monolithic (hard to use just the parts you need)
- Configuration-heavy (too many ways to configure the same thing)

OgbujiPT provides a lighter-weight alternative focused on knowledge banks, with clear boundaries and explicit control.

### Does this support GPU for locally-hosted models?

Yes! Make sure your LLM backend (Toolio, llama.cpp, text-generation-webui, etc.) is configured with GPU support. OgbujiPT works with any OpenAI-compatible API, so GPU acceleration is handled by your backend.

### What's with the crazy name?

Enh?! Yo mama! 😝 My surname is Ogbuji, so it's a bit of a pun. This is the notorious OGPT, ya feel me?
