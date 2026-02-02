#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# demo/unified-kb/simple_unified_demo.py
'''
Simple demonstration of UnifiedKB - OgbujiPT's unified knowledge base API.

UnifiedKB provides a single interface for managing multiple backend stores
(RAM, PostgreSQL, Qdrant, Onya graphs, etc.) and automatically aggregates
results across them.

This demo shows:
1. Setting up multiple backends (in-memory stores for simplicity)
2. Inserting content across all backends
3. Searching and aggregating results from multiple sources
4. Managing backends (enable/disable, introspection)

Requirements:
    uv pip install ogbujipt sentence-transformers

Run:
    python demo/unified-kb/simple_unified_demo.py
'''

import asyncio
from sentence_transformers import SentenceTransformer

from ogbujipt.memory.unified import UnifiedKB
from ogbujipt.store.ram import RAMDataDB


# Sample documents about programming languages
SAMPLE_DOCS = [
    ('Python is a high-level programming language known for its simplicity and readability. '
     'It excels in data science, machine learning, and web development.',
     {'language': 'Python', 'category': 'programming', 'popularity': 'high'}),

    ('JavaScript is the language of web browsers, also popular on the server side. '
     'It runs in browsers and on servers via Node.js.',
     {'language': 'JavaScript', 'category': 'programming', 'popularity': 'high'}),

    ('Rust is a systems programming language focused on safety, speed, and concurrency. '
     'It prevents memory errors without garbage collection.',
     {'language': 'Rust', 'category': 'programming', 'popularity': 'growing'}),

    ('Go (Golang) is designed for building scalable network services and cloud infrastructure. '
     'It offers simplicity, fast compilation, and built-in concurrency.',
     {'language': 'Go', 'category': 'programming', 'popularity': 'high'}),

    ('Swift is Apple\'s modern language for iOS, macOS, and other Apple platforms. '
     'It combines performance with developer-friendly syntax.',
     {'language': 'Swift', 'category': 'programming', 'popularity': 'medium'}),
]

ENCODER_MODEL = 'all-MiniLM-L6-v2'
GENERAL_COLLECTION_NAME = 'general_kb'
LANGUAGE_COLLECTION_NAME = 'language_kb'
CACHE_COLLECTION_NAME = 'cache_kb'

async def main():
    print('===', 'UnifiedKB Demo: Unified Knowledge Base API', '===')

    # Load embedding model (shared across backends)
    print('[1] Loading embedding model…')
    model = SentenceTransformer(ENCODER_MODEL)
    print('    ✓ Model loaded: ', ENCODER_MODEL, '\n')

    # Create multiple backend stores
    print('[2] Setting up backend stores…')

    # Backend 1: General programming knowledge
    backend1 = RAMDataDB(embedding_model=model, collection_name=GENERAL_COLLECTION_NAME)
    await backend1.setup()
    print('    ✓ Backend 1: General programming KB (in-memory)', GENERAL_COLLECTION_NAME, '\n')

    # Backend 2: Language-specific knowledge
    backend2 = RAMDataDB(embedding_model=model, collection_name=LANGUAGE_COLLECTION_NAME)
    await backend2.setup()
    print('    ✓ Backend 2: Language-specific KB (in-memory)', LANGUAGE_COLLECTION_NAME, '\n')

    # Backend 3: Cache/temporary storage
    backend3 = RAMDataDB(embedding_model=model, collection_name=CACHE_COLLECTION_NAME)
    await backend3.setup()
    print('    ✓ Backend 3: Cache KB (in-memory)', CACHE_COLLECTION_NAME, '\n')

    # Create UnifiedKB and register backends
    print('[3] Creating UnifiedKB and registering backends…')
    kb = UnifiedKB()

    # Register with different weights (importance for scoring)
    kb.add_backend('general', backend1, weight=1.5,
                  metadata={'type': 'vector', 'persistence': 'volatile', 'scope': 'general'})
    kb.add_backend('language', backend2, weight=1.2,
                  metadata={'type': 'vector', 'persistence': 'volatile', 'scope': 'specific'})
    kb.add_backend('cache', backend3, weight=0.8,
                  metadata={'type': 'vector', 'persistence': 'volatile', 'scope': 'temporary'})

    print(f'    ✓ Registered {len(kb.backends)} backends', '\n')

    # Display backend info
    print('[4] Backend information:')
    for name, info in kb.list_backends().items():
        print(f'    • {name}:')
        print(f'      - Weight: {info["weight"]}')
        print(f'      - Enabled: {info["enabled"]}')
        print(f'      - Scope: {info["metadata"]["scope"]}')
    print()

    # Insert documents - they go to all enabled backends by default
    print('[5] Inserting sample documents…')
    print(f'    Inserting {len(SAMPLE_DOCS)} documents about programming languages')

    for i, (content, metadata) in enumerate(SAMPLE_DOCS):
        result_ids = await kb.insert(content, metadata=metadata)
        print(f'    ✓ Document {i+1} inserted to {len(result_ids)} backends: {metadata["language"]}')
    print()

    # Search across all backends
    print('[6] Searching across all backends…')
    print('    Query: "web development and programming"', '\n')

    results = []
    async for result in kb.search('web development and programming', limit=5):
        results.append(result)

    print(f'    Found {len(results)} results (aggregated from all backends):')
    for i, result in enumerate(results, 1):
        # Extract language from metadata
        lang = result.metadata.get('language', 'UNKNOWN')
        print(f'    {i}. [{result.source}] Score: {result.score:.3f} | {lang}')
        print(f'       {result.content[:80]}…', '\n')

    # Search specific backends only
    print('[7] Searching specific backends…')
    print('    Query: "systems programming" (searching only "general" backend)', '\n')

    results = []
    async for result in kb.search('systems programming', backends=['general'], limit=3):
        results.append(result)

    print(f'    Found {len(results)} results:')
    for i, result in enumerate(results, 1):
        lang = result.metadata.get('language', 'UNKNOWN')
        print(f'    {i}. Score: {result.score:.3f} | {lang}')
        print(f'       {result.content[:80]}…', '\n')

    # Demonstrate backend enable/disable
    print('[8] Managing backends: disable cache backend')
    kb.disable_backend('cache')
    print('    ✓ Cache backend disabled', '\n')

    print('[9] Searching with cache disabled…')
    print('    Query: "mobile apps"', '\n')

    results = []
    async for result in kb.search('mobile apps', limit=3):
        results.append(result)

    print(f'    Found {len(results)} results (cache backend excluded):')
    for i, result in enumerate(results, 1):
        lang = result.metadata.get('language', 'UNKNOWN')
        print(f'    {i}. [{result.source}] Score: {result.score:.3f} | {lang}')
        print(f'       {result.content[:60]}…', '\n')

    # Re-enable cache
    print('[10] Re-enabling cache backend')
    kb.enable_backend('cache')
    print('     ✓ Cache backend re-enabled', '\n')

    # Insert to specific backends only
    print('[11] Selective insertion: adding document to cache only')
    new_doc = 'TypeScript adds static typing to JavaScript according to tooling and error detection preferences.'
    result_ids = await kb.insert(new_doc,
                                metadata={'language': 'TypeScript', 'category': 'programming'},
                                backends=['cache'])  # Only to cache

    print(f'     ✓ Inserted to: {list(result_ids.keys())}', '\n')

    # Cleanup
    print('[12] Cleaning up…')
    await backend1.cleanup()
    await backend2.cleanup()
    await backend3.cleanup()
    print('     ✓ All backends cleaned up', '\n')

    print('===', 'Demo complete!', '===', '\n')
    print('Key takeaways:')
    print('• UnifiedKB provides a single interface for multiple backends')
    print('• Results are automatically aggregated and sorted by score')
    print('• You can search all backends or specific ones')
    print('• Backends can be enabled/disabled without removal')
    print('• Insert operations can target all or specific backends')
    print('• Different backends can have different weights for scoring')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
