#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# demo/kgraph/simple_onya_demo.py
'''
Simple demonstration of OnyaKB - loading and searching .onya files.

This demo shows:
1. Loading .onya files from a directory
2. Text-based search across node properties
3. Type-based filtering (e.g., find all Person nodes)
4. Property-based search (e.g., find nodes with specific property values)
5. Retrieving individual nodes by IRI

Prerequisites:
    - OgbujiPT installed: `uv pip install -U .`
    - Sample .onya files in a directory

Run:
    python demo/kgraph/simple_onya_demo.py
'''

import asyncio
from pathlib import Path

from ogbujipt.store.kgraph import OnyaKB
from ogbujipt.retrieval.graph import TypeSearch, PropertySearch


# Path to sample .onya files (using test resources for this demo)
ONYA_FOLDER = Path(__file__).parent.parent.parent / 'test' / 'resource' / 'onya'


async def demo_basic_loading():
    '''Demonstrate loading .onya files'''
    print('='*70)
    print('DEMO 1: Loading .onya Files')
    print('='*70)

    kb = OnyaKB(folder_path=ONYA_FOLDER, name='demo_kb')
    await kb.setup()

    print(f'✓ Loaded {kb.count_nodes()} nodes from .onya files')
    print('\nFiles loaded:')
    for file_info in kb.get_loaded_files():
        print(f'  - {file_info["name"]} (document: {file_info["doc_iri"]})')

    await kb.cleanup()
    print()


async def demo_text_search():
    '''Demonstrate text-based search'''
    print('='*70)
    print('DEMO 2: Text-Based Search')
    print('='*70)

    kb = OnyaKB(folder_path=ONYA_FOLDER)
    await kb.setup()

    # Search for 'Python'
    print('\nSearching for "Python":')
    print('-'*70)
    count = 0
    async for result in kb.search('Python', limit=5):
        count += 1
        print(f'\nResult {count} (score: {result.score:.3f}):')
        print(f'  Node ID: {result.metadata["node_id"]}')
        # Print first few lines of content
        content_lines = result.content.split('\n')[:4]
        for line in content_lines:
            print(f'  {line}')
        if len(result.content.split('\n')) > 4:
            print('  ...')

    await kb.cleanup()
    print()


async def demo_type_search():
    '''Demonstrate type-based search using TypeSearch strategy'''
    print('='*70)
    print('DEMO 3: Type-Based Search')
    print('='*70)

    kb = OnyaKB(folder_path=ONYA_FOLDER)
    await kb.setup()

    # Find all Person nodes
    print('\nFinding all Person nodes:')
    print('-'*70)
    strategy = TypeSearch(type_iri='http://example.org/people/Person')

    count = 0
    async for result in strategy.execute('', backends=[kb], limit=10):
        count += 1
        # Extract name from content
        for line in result.content.split('\n'):
            if line.startswith('name:'):
                name = line.split(':', 1)[1].strip()
                print(f'{count}. {name} ({result.metadata["node_id"]})')
                break

    # Find all Organization nodes
    print('\nFinding all Organization nodes:')
    print('-'*70)
    org_strategy = TypeSearch(type_iri='http://example.org/companies/Organization')

    count = 0
    async for result in org_strategy.execute('', backends=[kb], limit=10):
        count += 1
        for line in result.content.split('\n'):
            if line.startswith('name:'):
                name = line.split(':', 1)[1].strip()
                print(f'{count}. {name} ({result.metadata["node_id"]})')
                break

    await kb.cleanup()
    print()


async def demo_property_search():
    '''Demonstrate property-based search'''
    print('='*70)
    print('DEMO 4: Property-Based Search')
    print('='*70)

    kb = OnyaKB(folder_path=ONYA_FOLDER)
    await kb.setup()

    # Find nodes with 'engineer' in their bio
    print('\nFinding nodes with "engineer" in bio:')
    print('-'*70)
    strategy = PropertySearch(
        property_label='bio',
        match_value='engineer',
        match_type='contains'
    )

    async for result in strategy.execute('', backends=[kb], limit=5):
        for line in result.content.split('\n'):
            if line.startswith('name:'):
                name = line.split(':', 1)[1].strip()
                print(f'  • {name}')
                break

    # Find exact occupation match
    print('\nFinding nodes with occupation = "Data Scientist":')
    print('-'*70)
    exact_strategy = PropertySearch(
        property_label='occupation',
        match_value='Data Scientist',
        match_type='equals'
    )

    async for result in exact_strategy.execute('', backends=[kb], limit=5):
        for line in result.content.split('\n'):
            if line.startswith('name:'):
                name = line.split(':', 1)[1].strip()
                print(f'  • {name}')
                break

    await kb.cleanup()
    print()


async def demo_node_retrieval():
    '''Demonstrate retrieving individual nodes'''
    print('='*70)
    print('DEMO 5: Individual Node Retrieval')
    print('='*70)

    kb = OnyaKB(folder_path=ONYA_FOLDER)
    await kb.setup()

    # Get a specific node
    node_iri = 'http://example.org/people/Alice'
    print(f'\nRetrieving node: {node_iri}')
    print('-'*70)

    node = await kb.get_node(node_iri)
    if node:
        print(f'ID: {node["id"]}')
        print(f'Types: {", ".join(node["types"])}')
        print('Properties:')
        for key, value in node['properties'].items():
            print(f'  {key}: {value}')
    else:
        print('Node not found')

    # Try to get a non-existent node
    missing_iri = 'http://example.org/people/NonExistent'
    print(f'\nRetrieving non-existent node: {missing_iri}')
    print('-'*70)
    missing_node = await kb.get_node(missing_iri)
    print(f'Result: {missing_node}')

    await kb.cleanup()
    print()


async def main():
    '''Run all demos'''
    print('\n')
    print('╔' + '='*68 + '╗')
    print('║' + ' '*15 + 'OnyaKB Demo - Knowledge Graph Search' + ' '*16 + '║')
    print('╚' + '='*68 + '╝')
    print()

    await demo_basic_loading()
    await demo_text_search()
    await demo_type_search()
    await demo_property_search()
    await demo_node_retrieval()

    print('='*70)
    print('Demo complete!')
    print('='*70)
    print()


if __name__ == '__main__':
    asyncio.run(main())
