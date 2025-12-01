# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.retrieval.graph
'''
Graph-based retrieval strategies for knowledge graphs.

Provides search strategies that leverage graph structure and relationships,
complementing vector-based search. Useful for GraphRAG applications.

Examples:
    Basic type-based retrieval:

    >>> from ogbujipt.retrieval.graph import TypeSearch
    >>> from ogbujipt.store.kgraph import OnyaKB
    >>> import asyncio
    >>>
    >>> async def example():
    ...     kb = OnyaKB(folder_path='./knowledge')
    ...     await kb.setup()
    ...
    ...     # Search for all Person nodes
    ...     type_search = TypeSearch(type_iri='http://schema.org/Person')
    ...     async for result in type_search.execute(
    ...         query='',  # Query not used for type search
    ...         backends=[kb],
    ...         limit=10
    ...     ):
    ...         print(result.content)
    >>>
    >>> asyncio.run(example())

See Also:
    - retrieval.dense - Dense vector search
    - retrieval.sparse - Sparse/BM25 search
    - retrieval.hybrid - Hybrid search strategies
'''

from typing import AsyncIterator

from ogbujipt.memory.base import SearchResult

__all__ = ['TypeSearch', 'PropertySearch']


class TypeSearch:
    '''
    Search strategy that retrieves nodes by type.

    Useful for filtering knowledge graphs by entity type (e.g., all Person nodes,
    all Organization nodes) as a complement to semantic search.

    Example:
        >>> from ogbujipt.retrieval.graph import TypeSearch
        >>> from ogbujipt.store.kgraph import OnyaKB
        >>>
        >>> kb = OnyaKB(folder_path='./knowledge')
        >>> await kb.setup()
        >>>
        >>> # Get all Person nodes
        >>> strategy = TypeSearch(type_iri='http://schema.org/Person')
        >>> async for result in strategy.execute('', backends=[kb], limit=10):
        ...     print(result.metadata['node_id'])
    '''

    def __init__(self, type_iri: str):
        '''
        Initialize type-based search

        Args:
            type_iri: IRI of the type to search for (e.g., 'http://schema.org/Person')
        '''
        self.type_iri = type_iri

    async def execute(
        self,
        query: str,
        backends: list,
        limit: int = 5,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''
        Execute type-based search across backends.

        Args:
            query: Query string (not used for type search, can be empty)
            backends: List of KB backends to search (must support search_by_type)
            limit: Maximum number of results per backend
            **kwargs: Additional search options

        Yields:
            SearchResult objects for nodes matching the type
        '''
        for backend in backends:
            # Check if backend supports type-based search
            if not hasattr(backend, 'search_by_type'):
                # Skip backends that don't support type search
                continue

            # Execute type search
            async for result in backend.search_by_type(
                type_iri=self.type_iri,
                limit=limit
            ):
                yield result


class PropertySearch:
    '''
    Search strategy that retrieves nodes by property value.

    Useful for exact or pattern-based matching on specific properties
    (e.g., all nodes with age > 30, all nodes with name containing 'Smith').

    Example:
        >>> from ogbujipt.retrieval.graph import PropertySearch
        >>> from ogbujipt.store.kgraph import OnyaKB
        >>>
        >>> kb = OnyaKB(folder_path='./knowledge')
        >>> await kb.setup()
        >>>
        >>> # Find nodes with 'Python' in their description
        >>> strategy = PropertySearch(
        ...     property_label='description',
        ...     match_value='Python',
        ...     match_type='contains'
        ... )
        >>> async for result in strategy.execute('', backends=[kb], limit=10):
        ...     print(result.content)

    Note:
        This is a basic implementation. More advanced property queries
        (comparisons, ranges, etc.) may require backend-specific support.
    '''

    def __init__(
        self,
        property_label: str,
        match_value: str,
        match_type: str = 'contains'
    ):
        '''
        Initialize property-based search

        Args:
            property_label: Name/label of the property to search
            match_value: Value to match against
            match_type: Type of matching - 'contains', 'equals', 'startswith', 'endswith'
        '''
        self.property_label = property_label
        self.match_value = match_value
        self.match_type = match_type

        if match_type not in {'contains', 'equals', 'startswith', 'endswith'}:
            raise ValueError(
                f'Invalid match_type: {match_type}. '
                f'Must be one of: contains, equals, startswith, endswith'
            )

    def _matches(self, value: str) -> bool:
        '''Check if a property value matches the search criteria'''
        value_str = str(value).lower()
        match_str = str(self.match_value).lower()

        if self.match_type == 'contains':
            return match_str in value_str
        elif self.match_type == 'equals':
            return value_str == match_str
        elif self.match_type == 'startswith':
            return value_str.startswith(match_str)
        elif self.match_type == 'endswith':
            return value_str.endswith(match_str)

        return False

    async def execute(
        self,
        query: str,
        backends: list,
        limit: int = 5,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''
        Execute property-based search across backends.

        Args:
            query: Query string (not used for property search, can be empty)
            backends: List of KB backends to search
            limit: Maximum number of results per backend
            **kwargs: Additional search options

        Yields:
            SearchResult objects for nodes matching the property criteria

        Note:
            This implementation iterates through all nodes. For large graphs,
            consider using backend-specific optimized property queries.
        '''
        for backend in backends:
            # Check if backend is an OnyaKB with graph access
            if not hasattr(backend, '_graph') or not backend._initialized:
                continue

            results = []

            # Iterate through all nodes in the graph
            for node in backend._graph.values():
                matched = False

                # Check properties
                for prop in node.properties:
                    if prop.label == self.property_label:
                        if self._matches(prop.value):
                            matched = True
                            break

                if matched:
                    # Build content representation
                    content_parts = [f'Node: {node.id}']

                    if node.types:
                        types_str = ', '.join(str(t) for t in node.types)
                        content_parts.append(f'Types: {types_str}')

                    for prop in node.properties:
                        content_parts.append(f'{prop.label}: {prop.value}')

                    content_text = '\n'.join(content_parts)

                    results.append(SearchResult(
                        content=content_text,
                        score=1.0,  # Exact property match
                        metadata={
                            'node_id': str(node.id),
                            'types': [str(t) for t in node.types],
                            'matched_property': self.property_label,
                            'matched_value': self.match_value
                        },
                        source=backend.name
                    ))

            # Apply limit
            if limit > 0:
                results = results[:limit]

            # Yield results
            for result in results:
                yield result
