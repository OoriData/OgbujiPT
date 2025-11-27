# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.store.memory
'''
In-memory vector store implementations for testing and lightweight use cases.

These implementations provide the same interface as their PostgreSQL counterparts
but use in-memory data structures. Perfect for:
- Unit testing without external dependencies
- Prototyping and demos
- Small-scale applications
- Embedded use cases where persistence isn't required

Philosophy: Following hynek's "own your I/O boundaries" principle, these aren't
mocks - they're legitimate alternative implementations of the vector store interface.
'''

import json
from uuid import UUID
from datetime import datetime, timezone
from typing import Iterable, Callable, List, Sequence, AsyncIterator, Any
from collections import defaultdict

import numpy as np

from ogbujipt.memory.base import SearchResult
from ogbujipt.config import attr_dict

__all__ = ['InMemoryDataDB', 'InMemoryMessageDB']


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    '''Calculate cosine similarity between two vectors'''
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


class InMemoryDataDB:
    '''
    In-memory vector store for data/document snippets.

    Drop-in replacement for DataDB from ogbujipt.store.postgres.pgvector_data
    '''

    def __init__(self, embedding_model, collection_name: str, **kwargs):
        '''
        Initialize in-memory data store

        Args:
            embedding_model: Model with .encode() method (e.g., SentenceTransformer)
            collection_name: Name for this collection (replaces "table_name" terminology)
            **kwargs: Additional options for compatibility (ignored)
        '''
        self.collection_name = collection_name
        self._embedding_model = embedding_model
        self._items = []  # List of (embedding, content, metadata) tuples
        self._initialized = False

        # Calculate embedding dimension
        if hasattr(embedding_model, 'encode'):
            try:
                self._embed_dimension = len(self._embedding_model.encode(''))
            except Exception:
                self._embed_dimension = 0
        else:
            self._embed_dimension = 0

    # Lifecycle methods with abstracted names
    async def create_table(self) -> None:
        '''Initialize the collection (compatibility alias for setup)'''
        await self.setup()

    async def setup(self) -> None:
        '''Initialize the in-memory collection'''
        self._initialized = True
        self._items = []

    async def drop_table(self) -> None:
        '''Clean up the collection (compatibility alias for cleanup)'''
        await self.cleanup()

    async def cleanup(self) -> None:
        '''Clean up the in-memory collection'''
        self._initialized = False
        self._items = []

    async def table_exists(self) -> bool:
        '''Check if collection is initialized (compatibility alias for is_initialized)'''
        return await self.is_initialized()

    async def is_initialized(self) -> bool:
        '''Check if the collection is initialized'''
        return self._initialized

    async def count_items(self) -> int:
        '''Count the number of items in the collection'''
        return len(self._items)

    # Data operations
    async def insert(
            self,
            content: str,
            metadata: dict | None = None
    ) -> None:
        '''
        Insert a document into the collection

        Args:
            content: Text content of the document
            metadata: Optional metadata dictionary
        '''
        if not self._initialized:
            raise RuntimeError(f'Collection {self.collection_name} not initialized. Call setup() first.')

        # Get embedding
        content_embedding = self._embedding_model.encode(content)

        # Store as numpy array for efficient similarity computation
        embedding_array = np.array(content_embedding)

        # Store metadata as a copy to avoid external mutations
        metadata_copy = dict(metadata) if metadata else {}

        self._items.append((embedding_array, content, metadata_copy))

    async def insert_many(
            self,
            content_list: Iterable[tuple[str, dict]]
    ) -> None:
        '''
        Insert multiple documents into the collection

        Args:
            content_list: Iterable of (content, metadata) tuples
        '''
        for content, metadata in content_list:
            await self.insert(content, metadata)

    async def search(
            self,
            query: str,
            limit: int = 5,
            threshold: float | None = None,
            meta_filter: Callable | List[Callable] | None = None,
            **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''
        Semantic similarity search

        Args:
            query: Query string to search for
            limit: Maximum number of results (0 for unlimited)
            threshold: Minimum similarity score (0.0 to 1.0)
            meta_filter: Filter function(s) for metadata
            **kwargs: Additional options for compatibility

        Yields:
            SearchResult objects with content, score, metadata, and source
        '''
        if not self._initialized:
            raise RuntimeError(f'Collection {self.collection_name} not initialized. Call setup() first.')

        # Type validation
        if threshold is not None:
            if not isinstance(threshold, float) or not (0 <= threshold <= 1):
                raise TypeError('threshold must be a float between 0.0 and 1.0')
        if not isinstance(limit, int):
            raise TypeError('limit must be an integer')

        # Normalize meta_filter to a sequence
        meta_filter = meta_filter or ()
        if not isinstance(meta_filter, Sequence):
            meta_filter = (meta_filter,)

        # Get query embedding
        query_embedding = np.array(self._embedding_model.encode(query))

        # Calculate similarities and filter
        results = []
        for embedding, content, metadata in self._items:
            # Apply metadata filters
            if meta_filter:
                skip = False
                for filter_func in meta_filter:
                    if not callable(filter_func):
                        raise TypeError('All meta_filter items must be callable')

                    clause, expected_value = filter_func()
                    # Extract the key from the clause (simplified parsing)
                    # Format: "(metadata ->> 'key')::type = ${}"
                    import re
                    key_match = re.search(r"'([^']+)'", clause)
                    if not key_match:
                        continue

                    key = key_match.group(1)

                    # Check if key exists and matches
                    if key not in metadata:
                        skip = True
                        break

                    # Type coercion and comparison
                    actual_value = metadata[key]

                    # Handle ANY operator for match_oneof
                    if 'ANY' in clause:
                        if actual_value not in expected_value:
                            skip = True
                            break
                    else:
                        # Exact match
                        if actual_value != expected_value:
                            skip = True
                            break

                if skip:
                    continue

            # Calculate similarity
            similarity = _cosine_similarity(query_embedding, embedding)

            # Apply threshold
            if threshold is not None and similarity < threshold:
                continue

            results.append((similarity, content, metadata))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[0], reverse=True)

        # Apply limit
        if limit > 0:
            results = results[:limit]

        # Yield as SearchResult objects
        for similarity, content, metadata in results:
            yield SearchResult(
                content=content,
                score=float(similarity),
                metadata=dict(metadata),  # Return a copy
                source=self.collection_name
            )


class InMemoryMessageDB:
    '''
    In-memory vector store for messages/chat logs.

    Drop-in replacement for MessageDB from ogbujipt.store.postgres.pgvector_message
    '''

    def __init__(self, embedding_model, collection_name: str, window: int = 0, **kwargs):
        '''
        Initialize in-memory message store

        Args:
            embedding_model: Model with .encode() method (e.g., SentenceTransformer)
            collection_name: Name for this collection (replaces "table_name" terminology)
            window: Maximum number of messages to keep per history_key (0 for unlimited)
            **kwargs: Additional options for compatibility (ignored)
        '''
        self.collection_name = collection_name
        self._embedding_model = embedding_model
        self.window = window

        # Store messages grouped by history_key
        # Structure: {history_key: [(timestamp, role, content, embedding, metadata), ...]}
        self._messages = defaultdict(list)
        self._initialized = False

        # Calculate embedding dimension
        if hasattr(embedding_model, 'encode'):
            try:
                self._embed_dimension = len(self._embedding_model.encode(''))
            except Exception:
                self._embed_dimension = 0
        else:
            self._embed_dimension = 0

    @classmethod
    async def from_conn_params(cls, embedding_model, table_name, host=None, port=None,
                                db_name=None, user=None, password=None, window=0, **kwargs):
        '''
        Factory method for compatibility with PostgreSQL version.

        In-memory version ignores connection parameters.
        '''
        return cls(embedding_model, table_name, window=window, **kwargs)

    @classmethod
    async def from_conn_string(cls, conn_string, embedding_model, table_name, window=0, **kwargs):
        '''
        Factory method for compatibility with PostgreSQL version.

        In-memory version ignores connection string.
        '''
        return cls(embedding_model, table_name, window=window, **kwargs)

    # Lifecycle methods with abstracted names
    async def create_table(self) -> None:
        '''Initialize the collection (compatibility alias for setup)'''
        await self.setup()

    async def setup(self) -> None:
        '''Initialize the in-memory collection'''
        self._initialized = True
        self._messages.clear()

    async def drop_table(self) -> None:
        '''Clean up the collection (compatibility alias for cleanup)'''
        await self.cleanup()

    async def cleanup(self) -> None:
        '''Clean up the in-memory collection'''
        self._initialized = False
        self._messages.clear()

    async def table_exists(self) -> bool:
        '''Check if collection is initialized (compatibility alias for is_initialized)'''
        return await self.is_initialized()

    async def is_initialized(self) -> bool:
        '''Check if the collection is initialized'''
        return self._initialized

    async def count_items(self) -> int:
        '''Count total number of messages across all history keys'''
        return sum(len(msgs) for msgs in self._messages.values())

    def _apply_window(self, history_key: UUID) -> None:
        '''Apply windowing to keep only most recent N messages for a history_key'''
        if self.window > 0:
            messages = self._messages[history_key]
            # Sort by timestamp and keep only the most recent window messages
            messages.sort(key=lambda x: x[0])  # Sort by timestamp
            if len(messages) > self.window:
                self._messages[history_key] = messages[-self.window:]

    # Data operations
    async def insert(
            self,
            history_key: UUID | str,
            role: str,
            content: str,
            timestamp: datetime | None = None,
            metadata: dict | None = None
    ) -> None:
        '''
        Insert a message into the collection

        Args:
            history_key: UUID or string identifying the conversation/history
            role: Role of the message sender (e.g., 'user', 'assistant')
            content: Text content of the message
            timestamp: Message timestamp (defaults to now)
            metadata: Optional metadata dictionary
        '''
        if not self._initialized:
            raise RuntimeError(f'Collection {self.collection_name} not initialized. Call setup() first.')

        # Convert string to UUID if needed
        if not isinstance(history_key, UUID):
            history_key = UUID(history_key)

        if timestamp is None:
            timestamp = datetime.now(tz=timezone.utc)

        # Get embedding - ensure it's a numpy array
        content_embedding = self._embedding_model.encode(content)
        # Handle both numpy arrays and lists
        if not isinstance(content_embedding, np.ndarray):
            content_embedding = np.array(content_embedding)

        # Store metadata as a copy
        metadata_copy = dict(metadata) if metadata else {}

        # Add message
        self._messages[history_key].append(
            (timestamp, role, content, content_embedding, metadata_copy)
        )

        # Apply windowing
        self._apply_window(history_key)

    async def insert_many(
            self,
            content_list: Iterable[tuple[UUID | str, str, str, datetime, dict]]
    ) -> None:
        '''
        Insert multiple messages into the collection

        Args:
            content_list: Iterable of (history_key, role, content, timestamp, metadata) tuples
        '''
        for history_key, role, content, timestamp, metadata in content_list:
            await self.insert(history_key, role, content, timestamp, metadata)

    async def clear(self, history_key: UUID | str) -> None:
        '''
        Remove all messages for a given history_key

        Args:
            history_key: UUID or string identifying the conversation to clear
        '''
        if not isinstance(history_key, UUID):
            history_key = UUID(history_key)

        if history_key in self._messages:
            del self._messages[history_key]

    async def get_messages(
            self,
            history_key: UUID | str,
            since: datetime | None = None,
            limit: int = 0
    ):
        '''
        Retrieve messages for a given history_key

        Args:
            history_key: UUID or string identifying the conversation
            since: Only return messages after this timestamp
            limit: Maximum number of messages (0 for all)

        Returns:
            Generator of attr_dict objects with ts, role, content, metadata
        '''
        if not isinstance(history_key, UUID):
            history_key = UUID(history_key)

        if since is not None and not isinstance(since, datetime):
            raise TypeError('since must be a datetime or None')

        if not isinstance(limit, int):
            raise TypeError('limit must be an integer')

        messages = self._messages.get(history_key, [])

        # Filter by timestamp
        if since is not None:
            messages = [msg for msg in messages if msg[0] > since]

        # Sort by timestamp descending (most recent first)
        messages = sorted(messages, key=lambda x: x[0], reverse=True)

        # Apply limit
        if limit > 0:
            messages = messages[:limit]

        # Return generator (not yield from within async function)
        return (attr_dict({
            'ts': timestamp,
            'role': role,
            'content': content,
            'metadata': dict(metadata)  # Return a copy
        }) for timestamp, role, content, embedding, metadata in messages)

    async def search(
            self,
            history_key: UUID = None,
            text: str = None,
            since: datetime | None = None,
            threshold: float | None = None,
            limit: int = 1,
            **kwargs  # Accept additional keyword args for compatibility
    ):
        '''
        Semantic similarity search within a conversation

        Args:
            history_key: UUID identifying the conversation
            text: Query text to search for
            since: Only search messages after this timestamp
            threshold: Minimum similarity score (0.0 to 1.0)
            limit: Maximum number of results

        Returns:
            Generator of attr_dict objects with ts, role, content, metadata, cosine_similarity
        '''
        if not self._initialized:
            raise RuntimeError(f'Collection {self.collection_name} not initialized. Call setup() first.')

        # Type validation
        if threshold is not None:
            if not isinstance(threshold, float) or not (0 <= threshold <= 1):
                raise TypeError('threshold must be a float between 0.0 and 1.0')
        if not isinstance(limit, int):
            raise TypeError('limit must be an integer')
        if history_key is not None and not isinstance(history_key, UUID):
            history_key = UUID(history_key)
        if since is not None and not isinstance(since, datetime):
            raise TypeError('since must be a datetime or None')

        # Get query embedding
        query_embedding = self._embedding_model.encode(text)
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        messages = self._messages.get(history_key, [])

        # Filter by timestamp
        if since is not None:
            messages = [msg for msg in messages if msg[0] > since]

        # Calculate similarities
        results = []
        for timestamp, role, content, embedding, metadata in messages:
            similarity = _cosine_similarity(query_embedding, embedding)

            if threshold is None or similarity >= threshold:
                results.append((similarity, timestamp, role, content, metadata))

        # Sort by similarity descending
        results.sort(key=lambda x: x[0], reverse=True)

        # Apply limit
        if limit > 0:
            results = results[:limit]

        # Convert to attr_dict objects (generator) and return the generator itself
        gen = (
            attr_dict({
                'cosine_similarity': similarity,
                'ts': timestamp,
                'role': role,
                'content': content,
                'metadata': dict(metadata)
            })
            for similarity, timestamp, role, content, metadata in results
        )
        return gen
