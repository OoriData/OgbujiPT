#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# demo/memory-store/simple_search_demo.py
'''
In-Memory Vector Search Demo - Zero Setup Required!

Demonstrates vector search using in-memory storage - no database required.
Perfect for prototyping, learning, and small-scale applications.

Prerequisites:
    Install: uv pip install -U . && uv pip install sentence-transformers

Usage:
    python simple_search_demo.py
'''
# See pyproject.toml for instruction to ignore `E501` (line too long)

import asyncio
from sentence_transformers import SentenceTransformer

from ogbujipt.store import RAMDataDB

# Sample knowledge base documents about machine learning
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


async def setup_knowledge_base(embedding_model):
    '''Initialize in-memory knowledge base and populate with sample data'''
    print('\n📚 Setting up in-memory knowledge base…')

    # Create in-memory database (no PostgreSQL needed!)
    kb_db = RAMDataDB(
        embedding_model=embedding_model,
        collection_name='ml_knowledge'
    )

    # Initialize
    await kb_db.setup()
    print('✓ In-memory store initialized')

    # Populate with documents
    print(f'✓ Inserting {len(KNOWLEDGE_BASE)} documents…')
    for doc in KNOWLEDGE_BASE:
        await kb_db.insert(
            content=doc['content'],
            metadata=doc['metadata']
        )

    count = await kb_db.count_items()
    print(f'✓ Knowledge base ready with {count} documents\n')

    return kb_db


async def demonstrate_search(kb_db):
    '''Run example searches to demonstrate capabilities'''

    # Example queries
    queries = [
        'What is machine learning?',
        'How do neural networks work?',
        'Tell me about clustering algorithms',
        'What programming languages are used for AI?'
    ]

    print('🔍 Running Example Searches\n')
    print('=' * 80)

    for query in queries:
        print(f'\nQuery: "{query}"')
        print('-' * 80)

        # Search the knowledge base
        results = []
        async for result in kb_db.search(query, limit=3):
            results.append(result)

        # Display results
        for i, result in enumerate(results, 1):
            print(f'\n{i}. Score: {result.score:.3f} | {result.metadata.get("topic", "N/A")} ({result.metadata.get("difficulty", "N/A")})')
            print(f'   {result.content[:100]}...')

        print()

    print('=' * 80)


async def demonstrate_filtering(kb_db):
    '''Demonstrate metadata filtering'''
    from ogbujipt.store.postgres.pgvector import match_exact

    print('\n🎯 Filtered Search Demo\n')
    print('=' * 80)

    # Search only beginner-level content
    query = 'explain machine learning concepts'
    print(f'\nQuery: "{query}" (filtered to: difficulty=beginner)')
    print('-' * 80)

    beginner_filter = match_exact('difficulty', 'beginner')
    results = []
    async for result in kb_db.search(query, limit=5, meta_filter=beginner_filter):
        results.append(result)

    for i, result in enumerate(results, 1):
        print(f'\n{i}. Score: {result.score:.3f} | {result.metadata.get("topic", "N/A")}')
        print(f'   {result.content[:100]}...')

    print('\n' + '=' * 80)


async def main():
    '''Main demo function'''
    print('\n' + '=' * 80)
    print('  In-Memory Vector Search Demo')
    print('  No database setup required!')
    print('=' * 80)

    # Load embedding model (downloads automatically if needed)
    print('\n🤖 Loading embedding model (all-MiniLM-L6-v2)…')
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print('✓ Model loaded')

    # Setup knowledge base
    kb_db = await setup_knowledge_base(embedding_model)

    try:
        # Run demonstrations
        await demonstrate_search(kb_db)
        await demonstrate_filtering(kb_db)

        print('\n✨ Demo complete!')
        print('\nKey Takeaways:')
        print('  • No database setup required - pure Python + numpy')
        print('  • Instant startup - perfect for prototyping')
        print('  • Full-featured: search, filtering, metadata')
        print('  • Drop-in replacement for PostgreSQL-based stores')
        print('\nFor production/larger datasets, consider:')
        print('  • PostgreSQL with pgvector (see demo/pg-hybrid/)')
        print('  • Qdrant for dedicated vector search')

    finally:
        # Cleanup
        await kb_db.cleanup()
        print('\n✓ Cleanup complete')


if __name__ == '__main__':
    asyncio.run(main())
