# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.store.kgraph.onya_store
'''
In-memory knowledge graph storage using Onya format.

Loads .onya files from a directory and provides search/retrieval capabilities
following the KBBackend protocol. Ideal for static knowledge bases that don't
require persistence or frequent updates.

Philosophy: GraphRAG without the database overhead. Load your knowledge graph
from human-readable .onya files and search it in-memory.

Examples:
    Loading .onya files from a directory:

    >>> from ogbujipt.store.kgraph import OnyaKB
    >>> import asyncio
    >>>
    >>> async def example():
    ...     # Load all .onya files from a directory
    ...     kb = OnyaKB(folder_path='/path/to/onya/files')
    ...     await kb.setup()
    ...
    ...     # Search by text content (searches node properties)
    ...     async for result in kb.search('machine learning', limit=5):
    ...         print(f'{result.content} (score: {result.score:.3f})')
    ...
    ...     # Search by node type
    ...     async for result in kb.search_by_type('http://schema.org/Person', limit=10):
    ...         print(f'{result.content}')
    ...
    ...     await kb.cleanup()
    >>>
    >>> asyncio.run(example())

See Also:
    - demo/kgraph/ - Complete working examples
    - Onya documentation: https://github.com/OoriData/Onya
    - memory.base.KBBackend - Protocol interface
'''

from pathlib import Path
from typing import AsyncIterator, Any

from onya.graph import graph
from onya.serial import literate_lex

from ogbujipt.memory.base import SearchResult

__all__ = ['OnyaKB']


class OnyaKB:
    '''
    In-memory knowledge graph backend using Onya format.

    Loads .onya files from a folder and provides search capabilities over the
    resulting graph. Implements the KBBackend protocol for compatibility with
    OgbujiPT's unified KB system.

    Perfect for:
        - Static knowledge bases (ontologies, taxonomies, reference data)
        - GraphRAG applications without database overhead
        - Human-curated knowledge graphs
        - Embedded knowledge in applications

    Example:
        >>> from ogbujipt.store.kgraph import OnyaKB
        >>>
        >>> kb = OnyaKB(folder_path='./knowledge')
        >>> await kb.setup()
        >>>
        >>> # Text search across all node properties
        >>> async for result in kb.search('Python programming', limit=5):
        ...     print(result.content, result.score)
        >>>
        >>> # Type-based retrieval
        >>> async for result in kb.search_by_type('http://schema.org/Person'):
        ...     print(result.content)
        >>>
        >>> await kb.cleanup()

    See Also:
        store.ram.RAMDataDB - Alternative in-memory vector store
        memory.base.KBBackend - Protocol interface
    '''

    def __init__(self, folder_path: str | Path, name: str = 'onya_kb', **kwargs):
        '''
        Initialize Onya knowledge graph backend

        Args:
            folder_path: Directory containing .onya files to load
            name: Name for this KB instance (for logging/identification)
            **kwargs: Additional options for compatibility (currently unused)
        '''
        self.folder_path = Path(folder_path)
        self.name = name
        self._graph = None
        self._initialized = False
        self._loaded_files = []  # Track which files were loaded

    async def setup(self) -> None:
        '''
        Load all .onya files from the folder into the graph.

        Raises:
            FileNotFoundError: If folder_path doesn't exist
            ValueError: If no .onya files found in folder
        '''
        if not self.folder_path.exists():
            raise FileNotFoundError(f'Folder not found: {self.folder_path}')

        if not self.folder_path.is_dir():
            raise ValueError(f'Path is not a directory: {self.folder_path}')

        # Initialize empty graph
        self._graph = graph()

        # Find all .onya files
        onya_files = list(self.folder_path.glob('*.onya'))

        if not onya_files:
            raise ValueError(f'No .onya files found in {self.folder_path}')

        # Load each file
        for onya_file in onya_files:
            with open(onya_file, 'r', encoding='utf-8') as f:
                onya_text = f.read()

            # Parse into graph
            doc_iri = literate_lex.parse(onya_text, self._graph)
            self._loaded_files.append({
                'path': str(onya_file),
                'doc_iri': str(doc_iri) if doc_iri else None,
                'name': onya_file.name
            })

        self._initialized = True

    async def cleanup(self) -> None:
        '''Clean up the in-memory graph'''
        self._initialized = False
        if self._graph:
            self._graph.clear()
        self._graph = None
        self._loaded_files = []

    async def is_initialized(self) -> bool:
        '''Check if the KB is initialized'''
        return self._initialized

    def count_nodes(self) -> int:
        '''Count the number of nodes in the graph'''
        if not self._initialized or not self._graph:
            return 0
        return len(self._graph)

    def get_loaded_files(self) -> list[dict]:
        '''Get list of loaded .onya files with metadata'''
        return list(self._loaded_files)

    async def search(
        self,
        query: str,
        limit: int = 5,
        threshold: float | None = None,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''
        Search nodes by text content in properties.

        This is a simple text-based search that looks for the query string
        in node property values. For semantic search, combine with a vector
        store backend using hybrid search strategies.

        Args:
            query: Query string to search for
            limit: Maximum number of results to return (0 for unlimited)
            threshold: Minimum relevance score (0.0 to 1.0, currently unused)
            **kwargs: Additional search options

        Yields:
            SearchResult objects with matching nodes

        Note:
            This implementation does simple case-insensitive substring matching.
            For more advanced search, consider using with a vector embedding backend.
        '''
        if not self._initialized:
            raise RuntimeError(f'KB {self.name} not initialized. Call setup() first.')

        query_lower = query.lower()
        results = []

        # Search through all nodes and their properties
        for node in self._graph.values():
            # Build a text representation of the node
            content_parts = []

            # Add node ID
            node_id = str(node.id)
            content_parts.append(f'Node: {node_id}')

            # Add types
            if node.types:
                types_str = ', '.join(str(t) for t in node.types)
                content_parts.append(f'Types: {types_str}')

            # Add all properties
            property_matches = []
            for prop in node.properties:
                prop_value = str(prop.value)
                # Check for query match (case-insensitive)
                if query_lower in prop_value.lower():
                    property_matches.append(prop_value)
                content_parts.append(f'{prop.label}: {prop_value}')

            # Calculate simple relevance score based on matches
            content_text = '\n'.join(content_parts)
            match_count = content_text.lower().count(query_lower)

            if match_count > 0:
                # Simple scoring: normalize by content length
                score = min(1.0, match_count / (len(content_text) / 100))

                results.append((score, content_text, {
                    'node_id': node_id,
                    'types': [str(t) for t in node.types],
                    'match_count': match_count
                }))

        # Sort by score (descending)
        results.sort(key=lambda x: x[0], reverse=True)

        # Apply limit
        if limit > 0:
            results = results[:limit]

        # Yield as SearchResult objects
        for score, content, metadata in results:
            yield SearchResult(
                content=content,
                score=float(score),
                metadata=metadata,
                source=self.name
            )

    async def search_by_type(
        self,
        type_iri: str,
        limit: int = 0
    ) -> AsyncIterator[SearchResult]:
        '''
        Search for nodes of a specific type.

        Args:
            type_iri: IRI of the type to search for (e.g., 'http://schema.org/Person')
            limit: Maximum number of results (0 for unlimited)

        Yields:
            SearchResult objects for nodes of the specified type
        '''
        if not self._initialized:
            raise RuntimeError(f'KB {self.name} not initialized. Call setup() first.')

        matched_nodes = list(self._graph.typematch(type_iri))

        # Apply limit
        if limit > 0:
            matched_nodes = matched_nodes[:limit]

        # Yield results
        for node in matched_nodes:
            # Build content representation
            content_parts = [f'Node: {node.id}']

            # Add types
            if node.types:
                types_str = ', '.join(str(t) for t in node.types)
                content_parts.append(f'Types: {types_str}')

            # Add all properties
            for prop in node.properties:
                content_parts.append(f'{prop.label}: {prop.value}')

            content_text = '\n'.join(content_parts)

            yield SearchResult(
                content=content_text,
                score=1.0,  # Type matches are exact
                metadata={
                    'node_id': str(node.id),
                    'types': [str(t) for t in node.types],
                    'match_type': 'type_exact'
                },
                source=self.name
            )

    async def get_node(self, node_iri: str) -> dict | None:
        '''
        Retrieve a specific node by IRI.

        Args:
            node_iri: IRI of the node to retrieve

        Returns:
            Dictionary with node data, or None if not found
        '''
        if not self._initialized:
            raise RuntimeError(f'KB {self.name} not initialized. Call setup() first.')

        try:
            node = self._graph[node_iri]
        except KeyError:
            return None

        # Convert to dict representation
        return {
            'id': str(node.id),
            'types': [str(t) for t in node.types],
            'properties': {
                prop.label: prop.value
                for prop in node.properties
            }
        }

    async def insert(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        **kwargs
    ) -> Any:
        '''
        Insert is not supported for file-based Onya KB.

        Onya files are meant to be edited as text files and reloaded.
        This method is provided for KBBackend protocol compatibility but
        raises NotImplementedError.

        To add nodes, edit the .onya files directly and call setup() again.
        '''
        raise NotImplementedError(
            'Insert not supported for file-based OnyaKB. '
            'Edit .onya files directly and reload with setup().'
        )

    async def delete(
        self,
        item_id: Any,
        **kwargs
    ) -> bool:
        '''
        Delete is not supported for file-based Onya KB.

        Onya files are meant to be edited as text files and reloaded.
        This method is provided for KBBackend protocol compatibility but
        raises NotImplementedError.

        To remove nodes, edit the .onya files directly and call setup() again.
        '''
        raise NotImplementedError(
            'Delete not supported for file-based OnyaKB. '
            'Edit .onya files directly and reload with setup().'
        )
