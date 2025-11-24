#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# demo/pg-hybrid/hybrid_search_demo.py
'''
Hybrid Search Demo - Quick Start

Demonstrates combining dense vector search with sparse BM25 retrieval using
Reciprocal Rank Fusion (RRF) for superior search results.

Prerequisites:
    1. PostgreSQL with pgvector running (see README.md)
    2. Install: uv pip install -U . && uv pip install sentence-transformers

Usage:
    python hybrid_search_demo.py
'''

import asyncio
import os
from typing import AsyncIterator
from sentence_transformers import SentenceTransformer

from ogbujipt.store.postgres import DataDB
from ogbujipt.retrieval import BM25Search, HybridSearch, SimpleDenseSearch
from ogbujipt.memory.base import SearchResult


# Database connection parameters
PG_DB_NAME = os.environ.get('PG_DB_NAME', 'hybrid_demo')
PG_DB_HOST = os.environ.get('PG_DB_HOST', 'localhost')
PG_DB_PORT = int(os.environ.get('PG_DB_PORT', '5432'))
PG_DB_USER = os.environ.get('PG_DB_USER', 'demo_user')
PG_DB_PASSWORD = os.environ.get('PG_DB_PASSWORD', 'demo_pass_2025')

# Demo embedding model
DEMO_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

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
        table_name='ml_knowledge',
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
    print('   Created table: ml_knowledge')

    # Insert all documents
    await kb_db.insert_many([
        (doc['content'], doc['metadata'])
        for doc in KNOWLEDGE_BASE
    ])

    doc_count = await kb_db.count_items()
    print(f'   Inserted {doc_count} documents')

    return kb_db


async def demo_dense_search(kb_db, query):
    '''Demonstrate dense vector search (semantic similarity)'''
    print('\n🔍 DENSE VECTOR SEARCH (semantic similarity)')
    print('   Good at: conceptual similarity, synonyms')
    print('   Weak at: exact terminology, abbreviations\n')

    results = []
    async for result in kb_db.search(query=query, limit=3):
        results.append(result)

    for i, result in enumerate(results, 1):
        print(f'   {i}. [{result.score:.3f}] {result.content[:80]}…')
        print(f'      Topic: {result.metadata.get("topic", "unknown")}')


async def demo_sparse_search(kb_db, query):
    '''Demonstrate sparse BM25 search (keyword-based)'''
    print('\n🔍 SPARSE BM25 SEARCH (keyword-based)')
    print('   Good at: exact keywords, names, abbreviations')
    print('   Weak at: semantic similarity, synonyms\n')

    # Initialize BM25. term frequency saturation: 1.5, document length normalization: 0.75, IDF floor: 0.25
    bm25 = BM25Search(k1=1.5, b=0.75, epsilon=0.25)

    results = []
    async for result in bm25.execute(query=query, backends=[kb_db], limit=3):
        results.append(result)

    for i, result in enumerate(results, 1):
        print(f'   {i}. [{result.score:.3f}] {result.content[:80]}…')
        print(f'      Topic: {result.metadata.get("topic", "unknown")}')

    return bm25


async def demo_hybrid_search(kb_db, query, bm25):
    '''Demonstrate hybrid search combining dense + sparse with RRF'''
    print('\n🔍 HYBRID SEARCH (Dense + Sparse with RRF)')
    print('   Combines strengths of both approaches. Uses Reciprocal Rank Fusion to merge results\n')

    # Create hybrid search with both strategies
    hybrid = HybridSearch(
        strategies=[
            SimpleDenseSearch(),  # Dense vector search
            bm25                  # Sparse BM25 search
        ],
        k=60  # RRF constant
    )

    results = []
    async for result in hybrid.execute(query=query, backends=[kb_db], limit=3):
        results.append(result)

    for i, result in enumerate(results, 1):
        print(f'   {i}. [{result.score:.3f}] {result.content[:80]}…')
        print(f'      Topic: {result.metadata.get("topic", "unknown")}')
        print(f'      Sources: {result.source}')

        # Show individual strategy ranks if available
        if 'rrf_ranks' in result.metadata:
            ranks_str = ', '.join([
                f'{strategy}:#{rank}'
                for strategy, rank, _ in result.metadata['rrf_ranks']
            ])
            print(f'      Ranks: {ranks_str}')

    return hybrid


async def compare_all_methods(kb_db, bm25, hybrid, query):
    '''Compare all three methods side by side'''
    print(f'\n\n{"="*80}')
    print(f'📊 COMPARISON TEST')
    print(f'{"="*80}')
    print(f'\nQuery: "{query}"')
    print('(This query uses terminology where exact keywords matter)\n')

    # Dense
    print('DENSE (may miss abbreviations):')
    dense_results = []
    async for r in kb_db.search(query=query, limit=2):
        dense_results.append(r)
    for i, r in enumerate(dense_results, 1):
        print(f'  {i}. [{r.score:.3f}] {r.content[:70]}…')

    # Sparse
    print('\nSPARSE (catches keywords):')
    sparse_results = []
    async for r in bm25.execute(query=query, backends=[kb_db], limit=2):
        sparse_results.append(r)
    for i, r in enumerate(sparse_results, 1):
        print(f'  {i}. [{r.score:.3f}] {r.content[:70]}…')

    # Hybrid
    print('\nHYBRID (best of both):')
    hybrid_results = []
    async for r in hybrid.execute(query=query, backends=[kb_db], limit=2):
        hybrid_results.append(r)
    for i, r in enumerate(hybrid_results, 1):
        print(f'  {i}. [{r.score:.3f}] {r.content[:70]}…')


async def main():
    '''Main demo flow'''
    print('='*80)
    print('OgbujiPT Hybrid Search Demo')
    print('='*80)
    print('\nThis demo shows how combining dense vector search with sparse BM25')
    print('retrieval produces better results than either method alone.')

    # Load embedding model
    print('\n📦 Loading embedding model (this may take a minute)…')
    embedding_model = SentenceTransformer(DEMO_EMBEDDING_MODEL)
    print('   Model loaded!')

    # Setup database
    kb_db = await setup_database(embedding_model)

    # Demo query
    query1 = 'What are ML algorithms?'

    print(f'\n\n{"="*80}')
    print(f'TEST 1: Basic Query')
    print(f'{"="*80}')
    print(f'\nQuery: "{query1}"\n')

    # Run each search method
    await demo_dense_search(kb_db, query1)
    bm25 = await demo_sparse_search(kb_db, query1)
    hybrid = await demo_hybrid_search(kb_db, query1, bm25)

    # Comparison test with terminology
    query2 = 'Tell me about CNNs for image processing'
    await compare_all_methods(kb_db, bm25, hybrid, query2)

    # Summary
    print(f'\n\n{"="*80}')
    print('✅ Demo Complete!')
    print('='*80)
    print('\nKey Takeaways:')
    print('  • Dense search: Best for semantic/conceptual queries')
    print('  • Sparse BM25: Best for exact keywords and terminology')
    print('  • Hybrid RRF: Combines both for superior results')
    print('\nNext Steps:')
    print('  • Try hybrid_search_advanced.ipynb for interactive exploration')
    print('  • Run chat_with_hybrid_kb.py for conversational AI demo')
    print('  • Modify queries and parameters to see different results')

    # Cleanup
    print('\n🧹 Cleaning up…')
    await kb_db.drop_table()
    print('   Dropped demo table')

    print('\nDemo complete.')


if __name__ == '__main__':
    asyncio.run(main())
