# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.retrieval.dense
'''
Dense vector retrieval strategies.

Dense vector search uses learned embeddings to find semantically similar content.
This module provides strategies that work with backends supporting dense vector search.
'''

from typing import AsyncIterator

from ogbujipt.memory.base import SearchResult, KBBackend  # , SearchStrategy


class SimpleDenseSearch:
    '''
    Simple wrapper to make DataDB (and other dense vector backends) compatible
    with the SearchStrategy protocol.

    This strategy delegates to the backend's native dense vector search,
    making it easy to use with HybridSearch alongside sparse strategies like BM25.

    Example:
        >>> from ogbujipt.retrieval.dense import SimpleDenseSearch
        >>> from ogbujipt.retrieval.sparse import BM25Search
        >>> from ogbujipt.retrieval.hybrid import HybridSearch
        >>> from ogbujipt.store.postgres import DataDB
        >>>
        >>> # Create hybrid search combining dense and sparse
        >>> hybrid = HybridSearch(strategies=[
        ...     SimpleDenseSearch(),  # Dense vector search
        ...     BM25Search()          # Sparse BM25 search
        ... ])
        >>> results = hybrid.execute('machine learning', backends=[db], limit=10)
        >>> async for result in results:
        ...     print(f'{result.score:.3f}: {result.content[:50]}...')

    Note: This strategy simply forwards to the backend's search method.
    For backends that support dense vector search (like DataDB), this will
    use their native embedding-based similarity search.
    '''

    async def execute(
        self,
        query: str,
        backends: list[KBBackend],
        limit: int = 5,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''
        Execute dense vector search across backends.

        Args:
            query: Search query string (will be embedded by backend)
            backends: List of KB backends to search
            limit: Maximum number of results to return per backend
            **kwargs: Additional options passed to backend.search()

        Yields:
            SearchResult objects from dense vector search
        '''
        for backend in backends:
            async for result in backend.search(query=query, limit=limit, **kwargs):
                # Backends return SearchResult objects directly
                yield result


__all__ = ['SimpleDenseSearch']

