# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/store/test_qdrant.py
'''
Integration tests for Qdrant vector store.

By default, tests use in-memory Qdrant (no external dependencies).
To run integration tests against a real Qdrant instance:

pytest test/store/test_qdrant.py -m integration --run-integration

Or set environment variables:
- QDRANT_HOST (default: localhost)
- QDRANT_PORT (default: 6333)
- QDRANT_API_KEY (optional, for cloud instances)
'''

import pytest
import numpy as np
# from unittest.mock import MagicMock

from ogbujipt.store.qdrant.collection import collection
from ogbujipt.text.splitter import text_split_fuzzy


class MockEmbeddingModel:
    '''
    Mock embedding model that returns consistent embeddings for testing.
    Uses simple deterministic embeddings based on text hash.
    '''
    def __init__(self, vector_size=384):
        self.vector_size = vector_size
    
    def encode(self, text, **kwargs):
        '''
        Generate a deterministic embedding vector based on text content.
        For single strings, returns a numpy array.
        '''
        import hashlib
        
        if isinstance(text, str):
            # Create deterministic embedding from text hash
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            # Generate vector_size values deterministically
            np.random.seed(hash_val % (2**32))
            embedding = np.random.rand(self.vector_size).astype(np.float32)
            # Normalize to unit vector for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        else:
            # Handle list of strings
            return np.array([self.encode(t) for t in text])


@pytest.fixture
def embedding_model():
    '''Fixture providing a mock embedding model'''
    return MockEmbeddingModel(vector_size=384)


@pytest.fixture
def qdrant_collection(request, embedding_model):
    '''
    Fixture providing a Qdrant collection using in-memory storage.
    Collection is automatically cleaned up after test.
    '''
    collection_name = f'test_{request.node.name.lower()}'
    coll = collection(name=collection_name, embedding_model=embedding_model)
    
    yield coll
    
    # Cleanup: reset collection if it was initialized
    if coll._db_initialized:
        try:
            coll.reset()
        except Exception:
            pass  # Collection may already be deleted


@pytest.fixture
def qdrant_collection_integration(request, embedding_model):
    '''
    Integration test fixture for real Qdrant instance.
    Requires QDRANT_HOST environment variable or defaults to localhost:6333
    '''
    import os
    from ogbujipt.store import qdrant
    
    if not qdrant.QDRANT_AVAILABLE:
        pytest.skip('Qdrant client not available. Install with: pip install qdrant-client')
    
    from qdrant_client import QdrantClient
    
    host = os.environ.get('QDRANT_HOST', 'localhost')
    port = int(os.environ.get('QDRANT_PORT', '6333'))
    api_key = os.environ.get('QDRANT_API_KEY', None)
    
    collection_name = f'test_{request.node.name.lower()}'
    
    try:
        # Try to connect to real Qdrant instance
        if api_key:
            client = QdrantClient(url=f'http://{host}:{port}', api_key=api_key)
        else:
            client = QdrantClient(host=host, port=port)
        
        # Test connection
        client.get_collections()
        
        coll = collection(name=collection_name, embedding_model=embedding_model, db=client)
        
        yield coll
        
        # Cleanup
        if coll._db_initialized:
            try:
                coll.reset()
            except Exception:
                pass
    except Exception as e:
        pytest.skip(f'Could not connect to Qdrant instance at {host}:{port}: {e}')


def test_qdrant_collection_creation(qdrant_collection):
    '''Test that a Qdrant collection can be created'''
    assert qdrant_collection.name is not None
    assert qdrant_collection.db is not None
    assert not qdrant_collection._db_initialized  # Not initialized until first update


def test_qdrant_update_and_count(COME_THUNDER_POEM, qdrant_collection):
    '''Test updating a collection with text chunks and counting items'''
    # Split the poem into chunks
    chunks = list(text_split_fuzzy(
        COME_THUNDER_POEM,
        chunk_size=21,
        chunk_overlap=3,
        separator='\n'
    ))
    
    # Initially, collection should be empty
    assert qdrant_collection.count() == 0
    
    # Update with chunks
    qdrant_collection.update(chunks)
    
    # Collection should now have items
    assert qdrant_collection.count() == len(chunks)
    assert qdrant_collection._db_initialized


def test_qdrant_update_with_metadata(COME_THUNDER_POEM, qdrant_collection):
    '''Test updating a collection with metadata'''
    chunks = list(text_split_fuzzy(
        COME_THUNDER_POEM,
        chunk_size=21,
        chunk_overlap=3,
        separator='\n'
    ))
    
    # Create metadata for each chunk
    metas = [{'chunk_index': i, 'source': 'test_poem'} for i in range(len(chunks))]
    
    qdrant_collection.update(chunks, metas=metas)
    
    assert qdrant_collection.count() == len(chunks)


def test_qdrant_search(COME_THUNDER_POEM, qdrant_collection):
    '''Test searching the collection'''
    chunks = list(text_split_fuzzy(
        COME_THUNDER_POEM,
        chunk_size=21,
        chunk_overlap=3,
        separator='\n'
    ))
    
    qdrant_collection.update(chunks)
    
    # Search for something related to the poem
    query = 'thunder and lightning'
    results = qdrant_collection.search(query, limit=3)
    
    assert results is not None
    # query_points returns a QueryResponse object with a 'points' attribute
    if hasattr(results, 'points'):
        points = results.points
    else:
        # Fallback for list-like results
        points = results
    
    assert len(points) > 0
    assert len(points) <= 3  # Respect limit
    
    # Check result structure (Qdrant returns ScoredPoint objects)
    result = points[0]
    assert hasattr(result, 'score') or hasattr(result, 'id')
    assert hasattr(result, 'payload') or hasattr(result, 'vector')


def test_qdrant_search_empty_collection(qdrant_collection):
    '''Test searching an empty collection returns empty results'''
    results = qdrant_collection.search('test query', limit=5)
    # query_points returns a QueryResponse object, check its points attribute
    if hasattr(results, 'points'):
        assert len(results.points) == 0
    else:
        assert results == [] or len(results) == 0


def test_qdrant_reset(COME_THUNDER_POEM, qdrant_collection):
    '''Test resetting a collection'''
    chunks = list(text_split_fuzzy(
        COME_THUNDER_POEM,
        chunk_size=21,
        chunk_overlap=3,
        separator='\n'
    ))
    
    qdrant_collection.update(chunks)
    assert qdrant_collection.count() > 0
    
    # Reset the collection
    qdrant_collection.reset()
    
    # Collection should be empty and uninitialized
    assert not qdrant_collection._db_initialized
    assert qdrant_collection.count() == 0


def test_qdrant_multiple_updates(COME_THUNDER_POEM, qdrant_collection):
    '''Test multiple updates to the same collection'''
    chunks1 = list(text_split_fuzzy(
        COME_THUNDER_POEM,
        chunk_size=30,
        chunk_overlap=5,
        separator='\n'
    ))
    
    qdrant_collection.update(chunks1)
    count1 = qdrant_collection.count()
    
    # Add more chunks
    chunks2 = list(text_split_fuzzy(
        COME_THUNDER_POEM,
        chunk_size=15,
        chunk_overlap=2,
        separator='\n'
    ))
    
    qdrant_collection.update(chunks2)
    count2 = qdrant_collection.count()
    
    # Should have more items after second update
    assert count2 > count1
    assert count2 == count1 + len(chunks2)


@pytest.mark.integration
def test_qdrant_integration_update_and_search(COME_THUNDER_POEM, qdrant_collection_integration):
    '''Integration test: Update and search with real Qdrant instance'''
    chunks = list(text_split_fuzzy(
        COME_THUNDER_POEM,
        chunk_size=21,
        chunk_overlap=3,
        separator='\n'
    ))
    
    qdrant_collection_integration.update(chunks)
    assert qdrant_collection_integration.count() == len(chunks)
    
    results = qdrant_collection_integration.search('thunder', limit=2)
    # Handle QueryResponse object
    if hasattr(results, 'points'):
        assert len(results.points) > 0
    else:
        assert len(results) > 0


if __name__ == '__main__':
    raise SystemExit('Attention! Run with pytest')
