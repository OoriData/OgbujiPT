# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/memory/test_unified.py
'''
Unit tests for UnifiedKB class.

Run with: pytest test/memory/test_unified.py
'''

import pytest
from typing import AsyncIterator, Any
from unittest.mock import AsyncMock, MagicMock

from ogbujipt.memory.unified import UnifiedKB, BackendInfo
from ogbujipt.memory.base import SearchResult, KBBackend


# Mock backend for testing
class MockBackend:
    '''Mock KBBackend implementation for testing'''

    def __init__(self, name: str, results: list[SearchResult] | None = None):
        self.name = name
        self.results = results or []
        self.inserted_items = []
        self.deleted_items = []
        self.search_called = False
        self.insert_called = False
        self.delete_called = False

    async def search(
        self,
        query: Any,
        limit: int = 5,
        threshold: float | None = None,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''Mock search that yields pre-configured results'''
        self.search_called = True
        for i, result in enumerate(self.results[:limit]):
            yield result

    async def insert(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        **kwargs
    ) -> str:
        '''Mock insert that tracks insertions'''
        self.insert_called = True
        item_id = f'{self.name}-{len(self.inserted_items)}'
        self.inserted_items.append((content, metadata))
        return item_id

    async def delete(
        self,
        item_id: Any,
        **kwargs
    ) -> bool:
        '''Mock delete that tracks deletions'''
        self.delete_called = True
        self.deleted_items.append(item_id)
        return True


class TestUnifiedKBBasics:
    '''Test basic UnifiedKB functionality: initialization, backend management'''

    def test_initialization(self):
        '''UnifiedKB initializes with empty backends'''
        kb = UnifiedKB()
        assert kb.backends == {}
        assert kb.list_backends() == {}

    def test_add_backend(self):
        '''Adding a backend registers it correctly'''
        kb = UnifiedKB()
        backend = MockBackend('test')

        kb.add_backend('test', backend, weight=1.5, metadata={'type': 'mock'})

        assert 'test' in kb.backends
        assert kb.backends['test'].name == 'test'
        assert kb.backends['test'].backend == backend
        assert kb.backends['test'].weight == 1.5
        assert kb.backends['test'].enabled is True
        assert kb.backends['test'].metadata == {'type': 'mock'}

    def test_add_backend_duplicate_raises(self):
        '''Adding duplicate backend name raises ValueError'''
        kb = UnifiedKB()
        backend1 = MockBackend('test1')
        backend2 = MockBackend('test2')

        kb.add_backend('duplicate', backend1)

        with pytest.raises(ValueError, match='already registered'):
            kb.add_backend('duplicate', backend2)

    def test_add_backend_defaults(self):
        '''Backend registration uses sensible defaults'''
        kb = UnifiedKB()
        backend = MockBackend('test')

        kb.add_backend('test', backend)  # No optional args

        assert kb.backends['test'].weight == 1.0
        assert kb.backends['test'].enabled is True
        assert kb.backends['test'].metadata == {}

    def test_remove_backend(self):
        '''Removing a backend unregisters it'''
        kb = UnifiedKB()
        backend = MockBackend('test')
        kb.add_backend('test', backend)

        assert kb.remove_backend('test') is True
        assert 'test' not in kb.backends

    def test_remove_nonexistent_backend(self):
        '''Removing nonexistent backend returns False'''
        kb = UnifiedKB()
        assert kb.remove_backend('nonexistent') is False

    def test_list_backends(self):
        '''list_backends returns correct info'''
        kb = UnifiedKB()
        backend1 = MockBackend('test1')
        backend2 = MockBackend('test2')

        kb.add_backend('b1', backend1, weight=1.0, metadata={'type': 'vector'})
        kb.add_backend('b2', backend2, weight=2.0, enabled=False, metadata={'type': 'graph'})

        backends = kb.list_backends()

        assert len(backends) == 2
        assert backends['b1']['weight'] == 1.0
        assert backends['b1']['enabled'] is True
        assert backends['b1']['metadata'] == {'type': 'vector'}
        assert backends['b2']['weight'] == 2.0
        assert backends['b2']['enabled'] is False
        assert backends['b2']['metadata'] == {'type': 'graph'}

    def test_enable_backend(self):
        '''enable_backend activates a disabled backend'''
        kb = UnifiedKB()
        backend = MockBackend('test')
        kb.add_backend('test', backend, enabled=False)

        assert kb.backends['test'].enabled is False
        assert kb.enable_backend('test') is True
        assert kb.backends['test'].enabled is True

    def test_enable_nonexistent_backend(self):
        '''enable_backend returns False for nonexistent backend'''
        kb = UnifiedKB()
        assert kb.enable_backend('nonexistent') is False

    def test_disable_backend(self):
        '''disable_backend deactivates an enabled backend'''
        kb = UnifiedKB()
        backend = MockBackend('test')
        kb.add_backend('test', backend, enabled=True)

        assert kb.backends['test'].enabled is True
        assert kb.disable_backend('test') is True
        assert kb.backends['test'].enabled is False

    def test_disable_nonexistent_backend(self):
        '''disable_backend returns False for nonexistent backend'''
        kb = UnifiedKB()
        assert kb.disable_backend('nonexistent') is False


class TestUnifiedKBSearch:
    '''Test UnifiedKB search functionality'''

    @pytest.mark.asyncio
    async def test_search_single_backend(self):
        '''Search with single backend returns results'''
        kb = UnifiedKB()
        results = [
            SearchResult(content='result 1', score=0.9, metadata={}, source='test'),
            SearchResult(content='result 2', score=0.8, metadata={}, source='test'),
        ]
        backend = MockBackend('test', results=results)
        kb.add_backend('test', backend)

        collected_results = []
        async for result in kb.search('query', limit=10):
            collected_results.append(result)

        assert len(collected_results) == 2
        assert collected_results[0].content == 'result 1'
        assert collected_results[1].content == 'result 2'
        assert backend.search_called is True

    @pytest.mark.asyncio
    async def test_search_multiple_backends(self):
        '''Search with multiple backends aggregates results'''
        kb = UnifiedKB()

        # Backend 1 with high scores
        results1 = [
            SearchResult(content='backend1-result1', score=0.95, metadata={}, source='backend1'),
            SearchResult(content='backend1-result2', score=0.85, metadata={}, source='backend1'),
        ]
        backend1 = MockBackend('backend1', results=results1)

        # Backend 2 with lower scores
        results2 = [
            SearchResult(content='backend2-result1', score=0.75, metadata={}, source='backend2'),
            SearchResult(content='backend2-result2', score=0.65, metadata={}, source='backend2'),
        ]
        backend2 = MockBackend('backend2', results=results2)

        kb.add_backend('backend1', backend1)
        kb.add_backend('backend2', backend2)

        collected_results = []
        async for result in kb.search('query', limit=10):
            collected_results.append(result)

        # Should get all 4 results, sorted by score descending
        assert len(collected_results) == 4
        assert collected_results[0].score == 0.95  # Highest score first
        assert collected_results[3].score == 0.65  # Lowest score last
        assert backend1.search_called is True
        assert backend2.search_called is True

    @pytest.mark.asyncio
    async def test_search_respects_limit(self):
        '''Search limit constrains total results'''
        kb = UnifiedKB()

        # Each backend returns 3 results
        results1 = [
            SearchResult(content=f'b1-r{i}', score=0.9 - i*0.1, metadata={}, source='b1')
            for i in range(3)
        ]
        results2 = [
            SearchResult(content=f'b2-r{i}', score=0.85 - i*0.1, metadata={}, source='b2')
            for i in range(3)
        ]

        kb.add_backend('b1', MockBackend('b1', results=results1))
        kb.add_backend('b2', MockBackend('b2', results=results2))

        collected_results = []
        async for result in kb.search('query', limit=3):  # Limit to 3
            collected_results.append(result)

        # Should get only top 3 results across both backends
        assert len(collected_results) == 3

    @pytest.mark.asyncio
    async def test_search_specific_backends(self):
        '''Search with backends parameter uses only specified backends'''
        kb = UnifiedKB()

        backend1 = MockBackend('b1', results=[
            SearchResult(content='b1-result', score=0.9, metadata={}, source='b1')
        ])
        backend2 = MockBackend('b2', results=[
            SearchResult(content='b2-result', score=0.9, metadata={}, source='b2')
        ])

        kb.add_backend('b1', backend1)
        kb.add_backend('b2', backend2)

        collected_results = []
        async for result in kb.search('query', backends=['b1']):  # Only b1
            collected_results.append(result)

        assert len(collected_results) == 1
        assert collected_results[0].content == 'b1-result'
        assert backend1.search_called is True
        assert backend2.search_called is False  # Should not be called

    @pytest.mark.asyncio
    async def test_search_disabled_backend_skipped(self):
        '''Search skips disabled backends'''
        kb = UnifiedKB()

        backend1 = MockBackend('b1', results=[
            SearchResult(content='b1-result', score=0.9, metadata={}, source='b1')
        ])
        backend2 = MockBackend('b2', results=[
            SearchResult(content='b2-result', score=0.9, metadata={}, source='b2')
        ])

        kb.add_backend('b1', backend1, enabled=True)
        kb.add_backend('b2', backend2, enabled=False)  # Disabled

        collected_results = []
        async for result in kb.search('query'):
            collected_results.append(result)

        assert len(collected_results) == 1
        assert collected_results[0].content == 'b1-result'
        assert backend1.search_called is True
        assert backend2.search_called is False

    @pytest.mark.asyncio
    async def test_search_no_backends(self):
        '''Search with no backends returns nothing'''
        kb = UnifiedKB()

        collected_results = []
        async for result in kb.search('query'):
            collected_results.append(result)

        assert len(collected_results) == 0

    @pytest.mark.asyncio
    async def test_search_nonexistent_backend(self):
        '''Search with nonexistent backend name is handled gracefully'''
        kb = UnifiedKB()
        backend = MockBackend('b1', results=[
            SearchResult(content='result', score=0.9, metadata={}, source='b1')
        ])
        kb.add_backend('b1', backend)

        # Request search on nonexistent backend
        collected_results = []
        async for result in kb.search('query', backends=['nonexistent']):
            collected_results.append(result)

        # Should get no results (nonexistent backend skipped)
        assert len(collected_results) == 0


class TestUnifiedKBInsert:
    '''Test UnifiedKB insert functionality'''

    @pytest.mark.asyncio
    async def test_insert_single_backend(self):
        '''Insert to single backend works'''
        kb = UnifiedKB()
        backend = MockBackend('test')
        kb.add_backend('test', backend)

        result = await kb.insert('test content', metadata={'key': 'value'})

        assert 'test' in result
        assert backend.insert_called is True
        assert len(backend.inserted_items) == 1
        assert backend.inserted_items[0][0] == 'test content'
        assert backend.inserted_items[0][1] == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_insert_multiple_backends(self):
        '''Insert to multiple backends works in parallel'''
        kb = UnifiedKB()
        backend1 = MockBackend('b1')
        backend2 = MockBackend('b2')

        kb.add_backend('b1', backend1)
        kb.add_backend('b2', backend2)

        result = await kb.insert('test content', metadata={'key': 'value'})

        assert 'b1' in result
        assert 'b2' in result
        assert backend1.insert_called is True
        assert backend2.insert_called is True

    @pytest.mark.asyncio
    async def test_insert_specific_backends(self):
        '''Insert with backends parameter uses only specified backends'''
        kb = UnifiedKB()
        backend1 = MockBackend('b1')
        backend2 = MockBackend('b2')

        kb.add_backend('b1', backend1)
        kb.add_backend('b2', backend2)

        result = await kb.insert('content', backends=['b1'])

        assert 'b1' in result
        assert 'b2' not in result
        assert backend1.insert_called is True
        assert backend2.insert_called is False

    @pytest.mark.asyncio
    async def test_insert_no_backends_raises(self):
        '''Insert with no backends raises ValueError'''
        kb = UnifiedKB()

        with pytest.raises(ValueError, match='No backends available'):
            await kb.insert('content')

    @pytest.mark.asyncio
    async def test_insert_disabled_backend_skipped(self):
        '''Insert skips disabled backends'''
        kb = UnifiedKB()
        backend1 = MockBackend('b1')
        backend2 = MockBackend('b2')

        kb.add_backend('b1', backend1, enabled=True)
        kb.add_backend('b2', backend2, enabled=False)

        result = await kb.insert('content')

        assert 'b1' in result
        assert 'b2' not in result


class TestUnifiedKBDelete:
    '''Test UnifiedKB delete functionality'''

    @pytest.mark.asyncio
    async def test_delete_single_backend(self):
        '''Delete from single backend works'''
        kb = UnifiedKB()
        backend = MockBackend('test')
        kb.add_backend('test', backend)

        result = await kb.delete({'test': 'item-123'})

        assert result['test'] is True
        assert backend.delete_called is True
        assert 'item-123' in backend.deleted_items

    @pytest.mark.asyncio
    async def test_delete_multiple_backends(self):
        '''Delete from multiple backends works in parallel'''
        kb = UnifiedKB()
        backend1 = MockBackend('b1')
        backend2 = MockBackend('b2')

        kb.add_backend('b1', backend1)
        kb.add_backend('b2', backend2)

        result = await kb.delete({'b1': 'item-1', 'b2': 'item-2'})

        assert result['b1'] is True
        assert result['b2'] is True
        assert backend1.delete_called is True
        assert backend2.delete_called is True

    @pytest.mark.asyncio
    async def test_delete_specific_backends(self):
        '''Delete with backends parameter uses only specified backends'''
        kb = UnifiedKB()
        backend1 = MockBackend('b1')
        backend2 = MockBackend('b2')

        kb.add_backend('b1', backend1)
        kb.add_backend('b2', backend2)

        result = await kb.delete({'b1': 'item-1', 'b2': 'item-2'}, backends=['b1'])

        assert 'b1' in result
        assert result['b1'] is True
        assert 'b2' not in result
        assert backend1.delete_called is True
        assert backend2.delete_called is False

    @pytest.mark.asyncio
    async def test_delete_missing_item_id(self):
        '''Delete without item_id for a backend returns False for that backend'''
        kb = UnifiedKB()
        backend = MockBackend('test')
        kb.add_backend('test', backend)

        # No item_id for 'test' backend - explicitly request deletion from 'test'
        result = await kb.delete({'other': 'id-123'}, backends=['test'])

        assert result['test'] is False
        assert backend.delete_called is False
