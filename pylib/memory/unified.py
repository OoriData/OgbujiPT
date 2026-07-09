# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.memory.unified
'''
Unified Knowledge Base API - single interface for multiple backends and retrieval strategies.

This module provides UnifiedKB, which orchestrates searches across multiple backend stores
(RAM, PostgreSQL, Qdrant, Onya graphs, etc.) and intelligently aggregates results.

Philosophy: Composability over monolith. UnifiedKB is a thin orchestration layer that
doesn't duplicate backend capabilities - it coordinates them.

Examples:
    Basic setup with multiple backends:

    >>> from ogbujipt.memory.unified import UnifiedKB
    >>> from ogbujipt.store.ram import RAMDataDB
    >>> from ogbujipt.store.postgres import DataDB
    >>> from sentence_transformers import SentenceTransformer
    >>> import asyncio
    >>>
    >>> async def example():
    ...     # Initialize backends
    ...     model = SentenceTransformer('all-MiniLM-L6-v2')
    ...     ram_store = RAMDataDB(embedding_model=model, collection_name='docs')
    ...     await ram_store.setup()
    ...
    ...     pg_store = await DataDB.from_conn_params(
    ...         host='localhost', port=5432, user='user', password='pass',
    ...         db_name='kb', embedding_model=model, table_name='documents'
    ...     )
    ...
    ...     # Create unified KB
    ...     kb = UnifiedKB()
    ...     kb.add_backend('memory', ram_store, weight=1.0,
    ...                   metadata={'type': 'vector', 'persistence': 'volatile'})
    ...     kb.add_backend('postgres', pg_store, weight=1.5,
    ...                   metadata={'type': 'vector', 'persistence': 'durable'})
    ...
    ...     # Insert content (goes to all backends by default)
    ...     await kb.insert('Machine learning is a subset of AI',
    ...                    metadata={'topic': 'ML'})
    ...
    ...     # Search across all backends with automatic aggregation
    ...     async for result in kb.search('artificial intelligence', limit=10):
    ...         print(f'{result.score:.3f} [{result.source}]: {result.content[:50]}')
    ...
    ...     # List registered backends
    ...     for name, info in kb.list_backends().items():
    ...         print(f'{name}: {info["metadata"]}')
    >>>
    >>> asyncio.run(example())

    Selective backend usage:

    >>> async def selective_example():
    ...     kb = UnifiedKB()
    ...     # ... setup backends as above ...
    ...
    ...     # Search only specific backends
    ...     async for result in kb.search('query', backends=['postgres']):
    ...         print(result.content)
    ...
    ...     # Insert to specific backends only
    ...     await kb.insert('New content', backends=['memory'], metadata={})
    >>>
    >>> asyncio.run(selective_example())

See Also:
    - memory.base - Protocol definitions for KBBackend
    - retrieval.* - Search strategy implementations
    - PLAN.md - Development roadmap and architecture details
'''

from typing import Any, AsyncIterator
from dataclasses import dataclass, field
import asyncio

import structlog

from ogbujipt.memory.base import SearchResult, KBBackend


logger = structlog.get_logger()


@dataclass
class BackendInfo:
    '''
    Metadata about a registered backend.

    Attributes:
        name: Unique identifier for this backend
        backend: The actual KBBackend implementation
        weight: Relative weight for result scoring (higher = more important)
        enabled: Whether this backend is currently active
        metadata: Additional backend metadata (type, capabilities, etc.)
    '''
    name: str
    backend: KBBackend
    weight: float = 1.0
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class UnifiedKB:
    '''
    Unified Knowledge Base - orchestrates multiple backends with a single interface.

    UnifiedKB provides a simple API for managing and searching across multiple KB backends.
    It handles backend registration, search orchestration, and result aggregation.

    The design is intentionally minimal - backends maintain their own connection pools,
    embedding models, and implementation details. UnifiedKB just coordinates them.

    Attributes:
        backends: Dict mapping backend names to BackendInfo objects
    '''

    def __init__(self):
        '''Initialize an empty UnifiedKB with no backends.'''
        self.backends: dict[str, BackendInfo] = {}
        logger.info('unified_kb.initialized')

    def add_backend(
        self,
        name: str,
        backend: KBBackend,
        weight: float = 1.0,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None
    ) -> None:
        '''
        Register a backend with this UnifiedKB.

        Args:
            name: Unique identifier for this backend (e.g., 'postgres', 'memory', 'graph')
            backend: KBBackend implementation (DataDB, RAMDataDB, OnyaKB, etc.)
            weight: Relative importance for scoring results (default: 1.0)
                   Higher weights boost scores from this backend in aggregation
            enabled: Whether this backend is active (default: True)
                    Disabled backends are kept but not used in searches
            metadata: Optional dict with backend info (type, capabilities, etc.)

        Raises:
            ValueError: If a backend with this name already exists

        Example:
            >>> kb = UnifiedKB()
            >>> kb.add_backend('main', pg_store, weight=1.5,
            ...                metadata={'type': 'vector', 'persistence': 'durable'})
            >>> kb.add_backend('cache', ram_store, weight=1.0,
            ...                metadata={'type': 'vector', 'persistence': 'volatile'})
        '''
        if name in self.backends:
            raise ValueError(f'Backend "{name}" already registered. Use remove_backend() first.')

        backend_info = BackendInfo(
            name=name,
            backend=backend,
            weight=weight,
            enabled=enabled,
            metadata=metadata or {}
        )
        self.backends[name] = backend_info

        logger.info('unified_kb.backend_added',
                   backend_name=name, weight=weight, enabled=enabled, metadata=metadata)

    def remove_backend(self, name: str) -> bool:
        '''
        Unregister a backend from this UnifiedKB.

        Args:
            name: Name of the backend to remove

        Returns:
            True if backend was removed, False if it didn't exist

        Note:
            This does NOT call cleanup() on the backend. The caller is responsible
            for proper backend lifecycle management.

        Example:
            >>> kb = UnifiedKB()
            >>> kb.add_backend('temp', temp_store)
            >>> # ... use it ...
            >>> kb.remove_backend('temp')
            True
            >>> await temp_store.cleanup()  # Caller's responsibility
        '''
        if name in self.backends:
            del self.backends[name]
            logger.info('unified_kb.backend_removed', backend_name=name)
            return True

        logger.warning('unified_kb.backend_not_found', backend_name=name)
        return False

    def list_backends(self) -> dict[str, dict[str, Any]]:
        '''
        Get information about all registered backends.

        Returns:
            Dict mapping backend names to their info dicts containing:
            - weight: Result scoring weight
            - enabled: Whether backend is active
            - metadata: Backend metadata dict

        Example:
            >>> kb = UnifiedKB()
            >>> kb.add_backend('pg', pg_store, weight=1.5,
            ...                metadata={'type': 'vector', 'persistence': 'durable'})
            >>> backends = kb.list_backends()
            >>> print(backends['pg']['weight'])  # 1.5
            >>> print(backends['pg']['metadata']['type'])  # 'vector'
        '''
        return {
            name: {
                'weight': info.weight,
                'enabled': info.enabled,
                'metadata': info.metadata
            }
            for name, info in self.backends.items()
        }

    def enable_backend(self, name: str) -> bool:
        '''
        Enable a disabled backend.

        Args:
            name: Name of backend to enable

        Returns:
            True if backend was enabled, False if not found

        Example:
            >>> kb.enable_backend('cache')
            True
        '''
        if name in self.backends:
            self.backends[name].enabled = True
            logger.info('unified_kb.backend_enabled', backend_name=name)
            return True
        return False

    def disable_backend(self, name: str) -> bool:
        '''
        Temporarily disable a backend without removing it.

        Disabled backends remain registered but won't be used in searches.
        Useful for maintenance or testing.

        Args:
            name: Name of backend to disable

        Returns:
            True if backend was disabled, False if not found

        Example:
            >>> kb.disable_backend('slow_backend')
            True
        '''
        if name in self.backends:
            self.backends[name].enabled = False
            logger.info('unified_kb.backend_disabled', backend_name=name)
            return True
        return False

    async def search(
        self,
        query: Any,
        limit: int = 10,
        backends: list[str] | None = None,
        threshold: float | None = None,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''
        Search across registered backends and aggregate results.

        Currently executes a simple parallel search across selected backends.
        Future versions will add query classification and intelligent routing.

        Args:
            query: Search query (typically str, but can be backend-specific)
            limit: Maximum total results to return (distributed across backends)
            backends: Optional list of backend names to search. If None, searches all enabled backends
            threshold: Optional minimum score threshold (backend-specific interpretation)
            **kwargs: Additional args passed to backend search methods

        Yields:
            SearchResult objects, aggregated and sorted by score (highest first)

        Example:
            >>> # Search all backends
            >>> async for result in kb.search('machine learning', limit=10):
            ...     print(f'{result.score:.3f}: {result.content}')
            >>>
            >>> # Search specific backends only
            >>> async for result in kb.search('query', backends=['postgres', 'graph']):
            ...     print(result.content)

        Note:
            This is a basic implementation. Phase 2B will add:
            - Query classification to select optimal backends/strategies
            - Score normalization across different backend types
            - Result deduplication
            - Optional re-ranking with cross-encoders
        '''
        # Determine which backends to search
        if backends is None:
            # Use all enabled backends
            target_backends = [
                (name, info.backend)
                for name, info in self.backends.items()
                if info.enabled
            ]
        else:
            # Use only specified backends (if they exist and are enabled)
            target_backends = []
            for name in backends:
                if name not in self.backends:
                    logger.warning('unified_kb.backend_not_found', backend_name=name)
                    continue
                info = self.backends[name]
                if not info.enabled:
                    logger.warning('unified_kb.backend_disabled', backend_name=name)
                    continue
                target_backends.append((name, info.backend))

        if not target_backends:
            logger.warning('unified_kb.no_backends_available')
            return

        logger.info('unified_kb.search_started',
                   query=str(query)[:100], limit=limit,
                   backend_count=len(target_backends))

        # Execute searches in parallel across all backends
        # Each backend gets the full limit - we'll aggregate later
        async def search_backend(name: str, backend: KBBackend) -> list[SearchResult]:
            '''Helper to collect results from one backend'''
            results = []
            try:
                async for result in backend.search(query, limit=limit, threshold=threshold, **kwargs):
                    results.append(result)
                logger.debug('unified_kb.backend_search_complete',
                           backend_name=name, result_count=len(results))
            except Exception as e:
                logger.error('unified_kb.backend_search_failed',
                           backend_name=name, error=str(e))
            return results

        # Launch all backend searches concurrently
        search_tasks = [
            search_backend(name, backend)
            for name, backend in target_backends
        ]
        all_results = await asyncio.gather(*search_tasks)

        # Flatten results from all backends
        combined_results = []
        for results in all_results:
            combined_results.extend(results)

        # Simple aggregation: sort by score descending, take top N
        # TODO Phase 2B: Add score normalization, deduplication, re-ranking
        combined_results.sort(key=lambda r: r.score, reverse=True)
        final_results = combined_results[:limit]

        logger.info('unified_kb.search_complete',
                   total_results=len(combined_results),
                   returned_results=len(final_results))

        # Yield results
        for result in final_results:
            yield result

    async def insert(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        backends: list[str] | None = None,
        **kwargs
    ) -> dict[str, Any]:
        '''
        Insert content into registered backends.

        Args:
            content: Text content to insert
            metadata: Optional metadata dict
            backends: Optional list of backend names. If None, inserts to all enabled backends
            **kwargs: Backend-specific insertion options

        Returns:
            Dict mapping backend names to their insertion results (backend-specific IDs)

        Raises:
            Exception: If all backend insertions fail

        Example:
            >>> result_ids = await kb.insert(
            ...     'Python is great for ML',
            ...     metadata={'topic': 'programming', 'lang': 'en'}
            ... )
            >>> print(result_ids)  # {'postgres': 123, 'memory': 'uuid-...'}

        Note:
            Insertions happen in parallel. If some backends fail, others may still succeed.
            The return dict will only contain successful insertions.
        '''
        # Determine target backends
        if backends is None:
            target_backends = [
                (name, info.backend)
                for name, info in self.backends.items()
                if info.enabled
            ]
        else:
            target_backends = []
            for name in backends:
                if name not in self.backends:
                    logger.warning('unified_kb.backend_not_found', backend_name=name)
                    continue
                info = self.backends[name]
                if not info.enabled:
                    logger.warning('unified_kb.backend_disabled', backend_name=name)
                    continue
                target_backends.append((name, info.backend))

        if not target_backends:
            raise ValueError('No backends available for insertion')

        logger.info('unified_kb.insert_started',
                   content_length=len(content),
                   backend_count=len(target_backends))

        # Insert in parallel to all backends
        async def insert_to_backend(name: str, backend: KBBackend) -> tuple[str, Any | None]:
            '''Helper to insert to one backend'''
            try:
                result = await backend.insert(content, metadata=metadata, **kwargs)
                logger.debug('unified_kb.backend_insert_complete', backend_name=name)
                return (name, result)
            except Exception as e:
                logger.error('unified_kb.backend_insert_failed',
                           backend_name=name, error=str(e))
                return (name, None)

        insert_tasks = [
            insert_to_backend(name, backend)
            for name, backend in target_backends
        ]
        results = await asyncio.gather(*insert_tasks)

        # Collect successful insertions
        result_map = {name: result for name, result in results if result is not None}

        if not result_map:
            raise Exception('All backend insertions failed')

        logger.info('unified_kb.insert_complete',
                   successful_backends=len(result_map),
                   failed_backends=len(target_backends) - len(result_map))

        return result_map

    async def delete(
        self,
        item_id: dict[str, Any],
        backends: list[str] | None = None,
        **kwargs
    ) -> dict[str, bool]:
        '''
        Delete content from registered backends.

        Args:
            item_id: Dict mapping backend names to their item IDs
                    (as returned by insert())
            backends: Optional list of backend names. If None, attempts deletion from all
            **kwargs: Backend-specific deletion options

        Returns:
            Dict mapping backend names to deletion success (True/False)

        Example:
            >>> # After insertion
            >>> ids = await kb.insert('content to delete')
            >>> # Delete from all backends
            >>> results = await kb.delete(ids)
            >>> print(results)  # {'postgres': True, 'memory': True}
        '''
        # Determine target backends
        if backends is None:
            target_backend_names = list(item_id.keys())
        else:
            target_backend_names = backends

        logger.info('unified_kb.delete_started',
                   item_ids=item_id,
                   backend_count=len(target_backend_names))

        # Delete in parallel from all backends
        async def delete_from_backend(name: str) -> tuple[str, bool]:
            '''Helper to delete from one backend'''
            if name not in self.backends:
                logger.warning('unified_kb.backend_not_found', backend_name=name)
                return (name, False)

            if name not in item_id:
                logger.warning('unified_kb.no_item_id', backend_name=name)
                return (name, False)

            try:
                backend = self.backends[name].backend
                success = await backend.delete(item_id[name], **kwargs)
                logger.debug('unified_kb.backend_delete_complete',
                           backend_name=name, success=success)
                return (name, success)
            except Exception as e:
                logger.error('unified_kb.backend_delete_failed',
                           backend_name=name, error=str(e))
                return (name, False)

        delete_tasks = [delete_from_backend(name) for name in target_backend_names]
        results = await asyncio.gather(*delete_tasks)

        result_map = dict(results)

        successful_count = sum(1 for success in result_map.values() if success)
        logger.info('unified_kb.delete_complete',
                   successful_backends=successful_count,
                   failed_backends=len(result_map) - successful_count)

        return result_map


__all__ = ['UnifiedKB', 'BackendInfo']
