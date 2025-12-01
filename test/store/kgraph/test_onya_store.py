# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test.store.kgraph.test_onya_store
'''
Tests for Onya-based knowledge graph storage
'''

import pytest
from pathlib import Path

from ogbujipt.store.kgraph import OnyaKB


# Path to test resources
TEST_RESOURCE_DIR = Path(__file__).parent.parent.parent / 'resource' / 'onya'


@pytest.mark.asyncio
async def test_onya_kb_setup():
    '''Test basic setup and loading of .onya files'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR, name='test_kb')

    # Initially not initialized
    assert not await kb.is_initialized()

    # Setup should load files
    await kb.setup()

    # Now initialized
    assert await kb.is_initialized()

    # Should have loaded 2 files
    loaded_files = kb.get_loaded_files()
    assert len(loaded_files) == 2

    # Should have multiple nodes
    node_count = kb.count_nodes()
    assert node_count > 0
    assert node_count == 5  # 3 people + 2 companies

    await kb.cleanup()


@pytest.mark.asyncio
async def test_onya_kb_setup_errors():
    '''Test error handling in setup'''
    # Test with non-existent folder
    kb = OnyaKB(folder_path='/nonexistent/folder')
    with pytest.raises(FileNotFoundError):
        await kb.setup()

    # Test with file instead of directory
    onya_file = TEST_RESOURCE_DIR / 'people.onya'
    kb = OnyaKB(folder_path=onya_file)
    with pytest.raises(ValueError, match='not a directory'):
        await kb.setup()


@pytest.mark.asyncio
async def test_onya_kb_search():
    '''Test text-based search in knowledge graph'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    # Search for 'Python'
    results = []
    async for result in kb.search('Python', limit=5):
        results.append(result)

    # Should find matches (Alice and DataSystems mention Python)
    assert len(results) > 0
    assert any('Python' in r.content for r in results)

    # Check result structure
    assert all(hasattr(r, 'content') for r in results)
    assert all(hasattr(r, 'score') for r in results)
    assert all(hasattr(r, 'metadata') for r in results)
    assert all(r.source == 'onya_kb' for r in results)

    await kb.cleanup()


@pytest.mark.asyncio
async def test_onya_kb_search_case_insensitive():
    '''Test that search is case-insensitive'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    # Search with different cases
    results_lower = []
    async for result in kb.search('machine learning', limit=5):
        results_lower.append(result)

    results_upper = []
    async for result in kb.search('MACHINE LEARNING', limit=5):
        results_upper.append(result)

    # Should get same number of results
    assert len(results_lower) == len(results_upper)
    assert len(results_lower) > 0

    await kb.cleanup()


@pytest.mark.asyncio
async def test_onya_kb_search_by_type():
    '''Test type-based search'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    # Search for Person type
    results = []
    async for result in kb.search_by_type('http://example.org/people/Person', limit=10):
        results.append(result)

    # Should find 3 people
    assert len(results) == 3
    # Types appear as full IRIs in content
    assert all('people/Person' in r.content or 'Type' in r.content for r in results)

    # Check metadata
    assert all('node_id' in r.metadata for r in results)
    assert all('types' in r.metadata for r in results)

    # Search for Organization type
    org_results = []
    async for result in kb.search_by_type('http://example.org/companies/Organization', limit=10):
        org_results.append(result)

    # Should find 2 organizations
    assert len(org_results) == 2
    assert all('companies/Organization' in r.content or 'Type' in r.content for r in org_results)

    await kb.cleanup()


@pytest.mark.asyncio
async def test_onya_kb_get_node():
    '''Test retrieving a specific node by IRI'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    # Get Alice node
    node = await kb.get_node('http://example.org/people/Alice')
    assert node is not None
    assert node['id'] == 'http://example.org/people/Alice'
    # Types are full IRIs
    types_list = node['types']
    assert any('Person' in t for t in types_list)
    assert 'name' in node['properties']
    assert node['properties']['name'] == 'Alice Smith'

    # Non-existent node
    missing = await kb.get_node('http://example.org/people/NonExistent')
    assert missing is None

    await kb.cleanup()


@pytest.mark.asyncio
async def test_onya_kb_insert_not_supported():
    '''Test that insert raises NotImplementedError'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    with pytest.raises(NotImplementedError, match='Insert not supported'):
        await kb.insert('some content', metadata={})

    await kb.cleanup()


@pytest.mark.asyncio
async def test_onya_kb_delete_not_supported():
    '''Test that delete raises NotImplementedError'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    with pytest.raises(NotImplementedError, match='Delete not supported'):
        await kb.delete('some_id')

    await kb.cleanup()


@pytest.mark.asyncio
async def test_onya_kb_search_limit():
    '''Test that limit parameter works correctly'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    # Search with limit
    results_limit_2 = []
    async for result in kb.search('engineer', limit=2):
        results_limit_2.append(result)

    assert len(results_limit_2) <= 2

    # Search with no limit
    results_no_limit = []
    async for result in kb.search('engineer', limit=0):
        results_no_limit.append(result)

    # No limit should return all matches
    assert len(results_no_limit) >= len(results_limit_2)

    await kb.cleanup()


@pytest.mark.asyncio
async def test_onya_kb_not_initialized():
    '''Test that operations fail when not initialized'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)

    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match='not initialized'):
        async for _ in kb.search('test'):
            pass

    with pytest.raises(RuntimeError, match='not initialized'):
        async for _ in kb.search_by_type('http://example.org/Person'):
            pass

    with pytest.raises(RuntimeError, match='not initialized'):
        await kb.get_node('http://example.org/people/Alice')
