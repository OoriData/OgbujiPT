# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test.retrieval.test_graph
'''
Tests for graph-based retrieval strategies
'''

import pytest
from pathlib import Path

from ogbujipt.retrieval.graph import TypeSearch, PropertySearch
from ogbujipt.store.kgraph import OnyaKB


# Path to test resources
TEST_RESOURCE_DIR = Path(__file__).parent.parent / 'resource' / 'onya'


@pytest.mark.asyncio
async def test_type_search():
    '''Test TypeSearch strategy'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    # Create TypeSearch for Person
    strategy = TypeSearch(type_iri='http://example.org/people/Person')

    # Execute search
    results = []
    async for result in strategy.execute('', backends=[kb], limit=10):
        results.append(result)

    # Should find 3 people
    assert len(results) == 3
    assert all('Person' in r.content for r in results)

    await kb.cleanup()


@pytest.mark.asyncio
async def test_type_search_with_limit():
    '''Test TypeSearch with limit'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    strategy = TypeSearch(type_iri='http://example.org/people/Person')

    # Execute with limit=2
    results = []
    async for result in strategy.execute('', backends=[kb], limit=2):
        results.append(result)

    # Should respect limit
    assert len(results) == 2

    await kb.cleanup()


@pytest.mark.asyncio
async def test_property_search_contains():
    '''Test PropertySearch with contains match'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    # Search for 'Python' in bio
    strategy = PropertySearch(
        property_label='bio',
        match_value='Python',
        match_type='contains'
    )

    results = []
    async for result in strategy.execute('', backends=[kb], limit=10):
        results.append(result)

    # Should find Alice (mentions Python in bio)
    assert len(results) > 0
    assert any('Alice' in r.content for r in results)

    await kb.cleanup()


@pytest.mark.asyncio
async def test_property_search_equals():
    '''Test PropertySearch with equals match'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    # Search for exact occupation
    strategy = PropertySearch(
        property_label='occupation',
        match_value='Software Engineer',
        match_type='equals'
    )

    results = []
    async for result in strategy.execute('', backends=[kb], limit=10):
        results.append(result)

    # Should find exactly Alice
    assert len(results) == 1
    assert 'Alice' in results[0].content

    await kb.cleanup()


@pytest.mark.asyncio
async def test_property_search_startswith():
    '''Test PropertySearch with startswith match'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    # Search for names starting with 'Bob'
    strategy = PropertySearch(
        property_label='name',
        match_value='Bob',
        match_type='startswith'
    )

    results = []
    async for result in strategy.execute('', backends=[kb], limit=10):
        results.append(result)

    # Should find Bob
    assert len(results) == 1
    assert 'Bob Jones' in results[0].content

    await kb.cleanup()


@pytest.mark.asyncio
async def test_property_search_case_insensitive():
    '''Test that PropertySearch is case-insensitive'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    # Search with different cases
    strategy_lower = PropertySearch(
        property_label='bio',
        match_value='python',
        match_type='contains'
    )

    strategy_upper = PropertySearch(
        property_label='bio',
        match_value='PYTHON',
        match_type='contains'
    )

    results_lower = []
    async for result in strategy_lower.execute('', backends=[kb], limit=10):
        results_lower.append(result)

    results_upper = []
    async for result in strategy_upper.execute('', backends=[kb], limit=10):
        results_upper.append(result)

    # Should get same results regardless of case
    assert len(results_lower) == len(results_upper)
    assert len(results_lower) > 0

    await kb.cleanup()


@pytest.mark.asyncio
async def test_property_search_invalid_match_type():
    '''Test that invalid match_type raises ValueError'''
    with pytest.raises(ValueError, match='Invalid match_type'):
        PropertySearch(
            property_label='name',
            match_value='test',
            match_type='invalid'
        )


@pytest.mark.asyncio
async def test_property_search_metadata():
    '''Test that PropertySearch results include proper metadata'''
    kb = OnyaKB(folder_path=TEST_RESOURCE_DIR)
    await kb.setup()

    strategy = PropertySearch(
        property_label='occupation',
        match_value='Data Scientist',
        match_type='equals'
    )

    results = []
    async for result in strategy.execute('', backends=[kb], limit=10):
        results.append(result)

    assert len(results) == 1
    result = results[0]

    # Check metadata structure
    assert 'node_id' in result.metadata
    assert 'types' in result.metadata
    assert 'matched_property' in result.metadata
    assert 'matched_value' in result.metadata
    assert result.metadata['matched_property'] == 'occupation'
    assert result.metadata['matched_value'] == 'Data Scientist'

    await kb.cleanup()
