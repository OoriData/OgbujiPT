# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.memory.base
'''
Base classes and interfaces for OgbujiPT knowledge bank (memory) system.

Philosophy: Minimal abstractions. Every layer must justify its existence.
We define protocols (structural subtyping) rather than rigid ABC hierarchies.
'''

from typing import Protocol, Any, AsyncIterator, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    '''
    A single search result from any KB backend.

    Attributes:
        content: The actual content/text of the result
        score: Similarity/relevance score (higher = more relevant)
        metadata: Additional metadata about the result
        source: Which backend produced this result (e.g., 'pgvector', 'qdrant', 'graph')
    '''
    content: str
    score: float
    metadata: dict[str, Any]
    source: str

    def __post_init__(self):
        '''Validate score is in reasonable range'''
        if not (0.0 <= self.score <= 1.0):
            # Some backends use different scoring, so just warn
            pass  # We'll normalize scores in result aggregation


class KBBackend(Protocol):
    '''
    Protocol defining the minimum interface for a knowledge base backend.

    Backends can be: vector stores (dense/sparse), graph databases, document stores, etc.
    Uses Protocol (PEP 544) for structural subtyping - no inheritance required.

    Example implementations:
    - DataDB from store.postgres.pgvector_data
    - MessageDB from store.postgres.pgvector_message
    - OnyaStore from store.graph.onya_store
    - QdrantCollection from store.qdrant.collection
    '''

    async def search(
        self,
        query: Any,  # Could be str, embedding vector, or structured query
        limit: int = 5,
        threshold: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''
        Search the knowledge base.

        Args:
            query: Search query (str for text, vector for semantic, dict for structured)
            limit: Maximum number of results to return
            threshold: Minimum score threshold (backend-specific)
            **kwargs: Backend-specific options (e.g., meta_filter for postgres)

        Yields:
            SearchResult objects
        '''
        ...

    async def insert(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        '''
        Insert an item into the knowledge base.

        Args:
            content: Text content to insert
            metadata: Optional metadata dict
            **kwargs: Backend-specific options

        Returns:
            Backend-specific identifier for the inserted item
        '''
        ...

    async def delete(
        self,
        item_id: Any,
        **kwargs
    ) -> bool:
        '''
        Delete an item from the knowledge base.

        Args:
            item_id: Identifier of item to delete (backend-specific format)
            **kwargs: Backend-specific options

        Returns:
            True if deleted, False if not found
        '''
        ...


class SearchStrategy(Protocol):
    '''
    Protocol for search/retrieval strategies.

    Strategies determine HOW to search (e.g., dense vector, sparse BM25, hybrid, graph traversal).
    Separate from backends which determine WHERE to search.
    '''

    async def execute(
        self,
        query: str,
        backends: list[KBBackend],
        limit: int = 5,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''
        Execute this search strategy across one or more backends.

        Args:
            query: User's search query
            backends: List of KB backends to search
            limit: Target number of results
            **kwargs: Strategy-specific options

        Yields:
            SearchResult objects (may be merged/reranked across backends)
        '''
        ...


__all__ = ['SearchResult', 'KBBackend', 'SearchStrategy']
