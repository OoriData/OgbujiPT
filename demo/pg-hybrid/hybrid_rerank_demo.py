#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# demo/pg-hybrid/hybrid_rerank_demo.py
'''
Hybrid Search + Reranking Demo

Demonstrates three-stage retrieval:
1. Dense vector search (semantic similarity)
2. Sparse BM25 search (keyword matching)
3. Cross-encoder reranking (final precise scoring)

This pattern combines the best of all approaches:
- Dense: Fast semantic similarity
- Sparse: Keyword precision
- Reranker: Accurate final ranking (but too slow for initial retrieval)

Prerequisites:
    1. PostgreSQL with pgvector running (see README.md)
    2. Install: uv pip install -U . && uv pip install sentence-transformers "rerankers[transformers]"

Usage:
    python hybrid_rerank_demo.py
'''

import asyncio
import os
from sentence_transformers import SentenceTransformer

from ogbujipt.store.postgres import DataDB
from ogbujipt.retrieval import BM25Search, HybridSearch, SimpleDenseSearch, RerankedHybridSearch

# Lazy import for rerankers (only needed when using reranking)
try:
    from rerankers import Reranker
    RERANKERS_AVAILABLE = True
except ImportError:
    RERANKERS_AVAILABLE = False


# Database connection parameters
PG_DB_NAME = os.environ.get('PG_DB_NAME', 'hybrid_demo')
PG_DB_HOST = os.environ.get('PG_DB_HOST', 'localhost')
PG_DB_PORT = int(os.environ.get('PG_DB_PORT', '5432'))
PG_DB_USER = os.environ.get('PG_DB_USER', 'demo_user')
PG_DB_PASSWORD = os.environ.get('PG_DB_PASSWORD', 'demo_pass_2025')

# Models
DEMO_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# Popular cross-encoder reranker models (pick one):
# - 'BAAI/bge-reranker-base': Good balance of speed and quality
# - 'BAAI/bge-reranker-large': Higher quality, slower
# - 'cross-encoder/ms-marco-MiniLM-L-12-v2': Faster, decent quality
# - 'zeroentropy/zerank-2': Instruction-following, multilingual (requires special config)
# To-do: Articulate support for the likes of https://huggingface.co/lightblue/lb-reranker-0.5B-v1.0
RERANKER_MODEL = 'BAAI/bge-reranker-base'
# RERANKER_MODEL = 'zeroentropy/zerank-2'  # e.g. uncomment to test ZeRank-2

# Model-specific configurations
RERANKER_CONFIGS = {
    'zeroentropy/zerank-2': {
        'model_kwargs': {
            'trust_remote_code': True,
        },
        'batch_size': 1,  # Process one at a time due to padding token issue
    },
    'BAAI/bge-reranker-base': {
        'batch_size': 32,
    },
    'BAAI/bge-reranker-large': {
        'batch_size': 16,
    },
}

# Sample knowledge base documents
KNOWLEDGE_BASE = [
    {
        'content': 'Machine learning (ML) is a subset of artificial intelligence that enables systems to learn from data without explicit programming.',
        'metadata': {'topic': 'ML basics', 'difficulty': 'beginner'}
    },
    {
        'content': 'Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes.',
        'metadata': {'topic': 'neural networks', 'difficulty': 'intermediate'}
    },
    {
        'content': 'Random forest is an ensemble learning algorithm that constructs multiple decision trees during training.',
        'metadata': {'topic': 'algorithms', 'difficulty': 'intermediate'}
    },
    {
        'content': 'Python is a high-level programming language widely used for ML development due to libraries like scikit-learn, TensorFlow, and PyTorch.',
        'metadata': {'topic': 'programming', 'difficulty': 'beginner'}
    },
    {
        'content': 'Gradient descent is an optimization algorithm used to minimize loss functions in ML by iteratively moving toward the minimum.',
        'metadata': {'topic': 'optimization', 'difficulty': 'intermediate'}
    },
    {
        'content': 'Supervised learning uses labeled training data to learn mappings from inputs to outputs. Examples include classification and regression.',
        'metadata': {'topic': 'ML basics', 'difficulty': 'beginner'}
    },
    {
        'content': 'Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images. They use convolutional layers.',
        'metadata': {'topic': 'deep learning', 'difficulty': 'advanced'}
    },
    {
        'content': 'K-means clustering is an unsupervised learning algorithm that partitions data into K clusters based on feature similarity.',
        'metadata': {'topic': 'clustering', 'difficulty': 'beginner'}
    },
    {
        'content': 'Transfer learning leverages pre-trained models on new tasks, reducing training time and data requirements significantly.',
        'metadata': {'topic': 'deep learning', 'difficulty': 'advanced'}
    },
    {
        'content': 'The backpropagation algorithm computes gradients of the loss function with respect to network weights using the chain rule.',
        'metadata': {'topic': 'neural networks', 'difficulty': 'advanced'}
    },
    {
        'content': 'Support Vector Machines (SVMs) find optimal hyperplanes that maximize the margin between different classes in the feature space.',
        'metadata': {'topic': 'algorithms', 'difficulty': 'intermediate'}
    },
    {
        'content': 'Overfitting occurs when a model learns training data too well, including noise, resulting in poor generalization to new data.',
        'metadata': {'topic': 'ML basics', 'difficulty': 'beginner'}
    },
]


async def setup_database(embedding_model):
    '''Initialize database connection and populate with sample data'''
    print('\nSetting up database…')

    # Connect to PostgreSQL
    kb_db = await DataDB.from_conn_params(
        embedding_model=embedding_model,
        table_name='ml_knowledge_rerank',
        db_name=PG_DB_NAME,
        host=PG_DB_HOST,
        port=PG_DB_PORT,
        user=PG_DB_USER,
        password=PG_DB_PASSWORD,
        itypes=['vector'],  # Create HNSW index for fast vector search
        ifuncs=['cosine']
    )

    # Drop existing table if present (for clean demo)
    if await kb_db.table_exists():
        await kb_db.drop_table()
        print('   Dropped existing table')

    # Create fresh table
    await kb_db.create_table()
    print('   Created table: ml_knowledge_rerank')

    # Insert all documents
    await kb_db.insert_many([
        (doc['content'], doc['metadata'])
        for doc in KNOWLEDGE_BASE
    ])

    doc_count = await kb_db.count_items()
    print(f'   Inserted {doc_count} documents')

    return kb_db


async def demo_comparison(kb_db, query, bm25, hybrid, reranked):
    '''Compare all four methods side by side'''
    print('\n\n', '='*80, '📊 COMPARISON: All Methods', '='*80, '\n')
    print(f'Query: "{query}"')
    print('Testing how each method ranks results for this query\n')

    # Dense only
    print('1️⃣  DENSE VECTOR SEARCH (semantic only):')
    dense_results = []
    async for r in kb_db.search(query=query, limit=3):
        dense_results.append(r)
    for i, r in enumerate(dense_results, 1):
        print(f'  {i}. [{r.score:.3f}] {r.content[:70]}…')

    # Sparse only
    print('\n2️⃣  SPARSE BM25 SEARCH (keyword only):')
    sparse_results = []
    async for r in bm25.execute(query=query, backends=[kb_db], limit=3):
        sparse_results.append(r)
    for i, r in enumerate(sparse_results, 1):
        print(f'  {i}. [{r.score:.3f}] {r.content[:70]}…')

    # Hybrid (RRF)
    print('\n3️⃣  HYBRID RRF (dense + sparse fusion):')
    hybrid_results = []
    async for r in hybrid.execute(query=query, backends=[kb_db], limit=3):
        hybrid_results.append(r)
    for i, r in enumerate(hybrid_results, 1):
        print(f'  {i}. [{r.score:.3f}] {r.content[:70]}…')

    # Reranked
    print('\n4️⃣  RERANKED HYBRID (RRF + cross-encoder):')
    reranked_results = []
    async for r in reranked.execute(query=query, backends=[kb_db], limit=3):
        reranked_results.append(r)
    for i, r in enumerate(reranked_results, 1):
        print(f'  {i}. [{r.score:.3f}] {r.content[:70]}…')
        if 'rrf_score' in r.metadata:
            print(f'      (RRF score: {r.metadata["rrf_score"]:.3f}, reranker boosted to {r.score:.3f})')


async def demo_reranking_impact(kb_db, query, hybrid, reranked):
    '''Show how reranking changes the order from RRF'''
    print('\n\n', '='*80, '🔄 RERANKING IMPACT', '='*80, '\n')
    print(f'Query: "{query}"')
    print('See how the cross-encoder reorders RRF results\n')

    # Get more results to show reordering
    print('BEFORE RERANKING (RRF):')
    hybrid_results = []
    async for r in hybrid.execute(query=query, backends=[kb_db], limit=5):
        hybrid_results.append(r)
    for i, r in enumerate(hybrid_results, 1):
        print(f'  {i}. [{r.score:.3f}] {r.content[:80]}…')

    print('\nAFTER RERANKING (Cross-Encoder):')
    reranked_results = []
    async for r in reranked.execute(query=query, backends=[kb_db], limit=5):
        reranked_results.append(r)
    for i, r in enumerate(reranked_results, 1):
        original_rank = r.metadata.get('original_rank', '?')
        print(f'  {i}. [{r.score:.3f}] {r.content[:80]}…')
        print(f'      (was rank #{original_rank} in RRF, reranker score: {r.score:.3f})')


async def main():
    '''Main demo flow'''
    print('\n', '='*80, 'OgbujiPT Hybrid Search + Reranking Demo', '='*80, '\n')
    print('This demo shows how adding a cross-encoder reranker improves hybrid search.')
    print('\nThree-stage retrieval:')
    print('  1. Fast initial retrieval with RRF (dense + sparse)')
    print('  2. Slow but accurate reranking of top candidates')
    print('  3. Final results ranked by cross-encoder relevance\n')

    if not RERANKERS_AVAILABLE:
        print('❌ ERROR: rerankers library not installed')
        print('Install with: uv pip install "rerankers[transformers]"')
        return

    # Load models
    print('📦 Loading models (this may take a minute)…')
    print(f'   Embedding model: {DEMO_EMBEDDING_MODEL}')
    embedding_model = SentenceTransformer(DEMO_EMBEDDING_MODEL)
    print('   ✓ Embedding model loaded')

    print(f'   Reranker model: {RERANKER_MODEL}')

    # Get model-specific config if available
    reranker_config = RERANKER_CONFIGS.get(RERANKER_MODEL, {})
    model_kwargs = reranker_config.get('model_kwargs', {})
    batch_size = reranker_config.get('batch_size')

    # Initialize reranker with appropriate config
    reranker_params = {'model_name': RERANKER_MODEL}
    if model_kwargs:
        reranker_params['model_kwargs'] = model_kwargs
    if batch_size:
        reranker_params['batch_size'] = batch_size

    reranker = Reranker(**reranker_params)
    print('   ✓ Reranker loaded')
    if batch_size == 1:
        print('   ⚠ Using batch_size=1 (slower but handles padding token issues)')

    # Setup database
    kb_db = await setup_database(embedding_model)

    # Create search strategies
    bm25 = BM25Search(k1=1.5, b=0.75, epsilon=0.25)

    hybrid = HybridSearch(
        strategies=[SimpleDenseSearch(), bm25],
        k=60
    )

    reranked = RerankedHybridSearch(
        strategies=[SimpleDenseSearch(), bm25],
        reranker=reranker,
        rerank_top_k=20,  # Rerank top 20 from initial retrieval
        k=60
    )

    # Demo 1: Compare all methods
    query1 = 'What are neural networks and how do they work?'
    await demo_comparison(kb_db, query1, bm25, hybrid, reranked)

    # Demo 2: Show reranking impact
    query2 = 'Tell me about optimization algorithms in machine learning'
    await demo_reranking_impact(kb_db, query2, hybrid, reranked)

    # Summary
    print('\n\n', '='*80, '✅ Demo Complete!', '='*80)
    print('\nKey Takeaways:')
    print('  • Dense search: Fast, semantic similarity')
    print('  • Sparse BM25: Fast, keyword precision')
    print('  • Hybrid RRF: Combines both (better than either alone)')
    print('  • Reranked: Most accurate (but slower due to cross-encoder)')
    print('\nBest Practice: Use hybrid RRF for initial retrieval, then rerank top-K')
    print('               This gives you speed + accuracy.')

    # Cleanup
    print('\nCleaning up…')
    await kb_db.drop_table()
    print('   Dropped demo table')

    print('\nDemo complete.')


if __name__ == '__main__':
    asyncio.run(main())
