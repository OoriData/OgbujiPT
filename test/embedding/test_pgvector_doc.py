# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/embedding/test_pgvector_doc.py
'''
After setup as described in the README.md for this directory, run the tests with:

pytest test

or, for just this test module:

pytest test/embedding/test_pgvector_doc.py

Uses fixtures from conftest.py in current & parent directories
'''

import pytest
# from unittest.mock import MagicMock, patch
from unittest.mock import MagicMock, DEFAULT  # noqa: F401

# from ogbujipt.embedding.pgvector import DocDB
import numpy as np

pacer_copypasta = [  # Demo document
    ('The FitnessGram™ Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as it'
    ' continues.'), 
    'The 20 meter pacer test will begin in 30 seconds. Line up at the start.', 
    'The running speed starts slowly, but gets faster each minute after you hear this signal.', 
    '[beep] A single lap should be completed each time you hear this sound.', 
    '[ding] Remember to run in a straight line, and run as long as possible.', 
    'The second time you fail to complete a lap before the sound, your test is over.', 
    'The test will begin on the word start. On your mark, get ready, start.'
]

# XXX: This is to get around the fact that we can't mock the SentenceTransformer class without importing it - Kai
class SentenceTransformer(object):
    def __init__(self, model_name_or_path):
        self.encode = MagicMock()


@pytest.mark.asyncio
async def test_PGv_embed_pacer(DB):
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])
    # Insert data
    for index, text in enumerate(pacer_copypasta):   # For each line in the copypasta
        await DB.insert(                            # Insert the line into the table
            content=text,                            # The text to be embedded
            title=f'Pacer Copypasta line {index}',   # Title metadata
            page_numbers=[1, 2, 3],                  # Page number metadata
            tags=['fitness', 'pacer', 'copypasta'],  # Tag metadata
        )

    assert await DB.count_items() == len(pacer_copypasta), Exception("Not all documents inserted")

    # search table with perfect match
    search_string = '[beep] A single lap should be completed each time you hear this sound.'
    sim_search = await DB.search(text=search_string, limit=3)
    assert sim_search is not None, Exception("No results returned from perfect search")

    await DB.drop_table()

@pytest.mark.asyncio
async def test_PGv_embed_many_pacer(DB):
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])
    # Insert data using insert_many()
    documents = (
        (
            text,
            ['fitness', 'pacer', 'copypasta'],
            f'Pacer Copypasta line {index}',
            [1, 2, 3]
        )
        for index, text in enumerate(pacer_copypasta)
    )
    await DB.insert_many(documents)

    assert await DB.count_items() == len(pacer_copypasta), Exception("Not all documents inserted")

    # Search table with perfect match
    search_string = '[beep] A single lap should be completed each time you hear this sound.'
    sim_search = await DB.search(text=search_string, limit=3)
    assert sim_search is not None, Exception("No results returned from perfect search")

    await DB.drop_table()


@pytest.mark.asyncio
async def test_PGv_search_filtered(DB):
    dummy_model = SentenceTransformer('mock_transformer')
    def encode_tweaker(*args, **kwargs):
        if args[0].startswith('Text'):
            return np.array([1, 2, 3])
        else:
            return np.array([100, 300, 500])

    dummy_model.encode.side_effect = encode_tweaker
    # Need to replace the default encoder set up by the fixture
    DB._embedding_model = dummy_model
    # Insert data
    for index, text in enumerate(pacer_copypasta):   # For each line in the copypasta
        await DB.insert(                            # Insert the line into the table
            content=text,                            # The text to be embedded
            title='Pacer Copypasta',   # Title metadata
            page_numbers=[index],                    # Page number metadata
            tags=['fitness', 'pacer', 'copypasta'],  # Tag metadata
        )

    assert await DB.count_items() == len(pacer_copypasta), Exception("Not all documents inserted")

    # search table with filtered match
    search_string = '[beep] A single lap should be completed each time you hear this sound.'
    sim_search = await DB.search(
        text=search_string,
        query_title='Pacer Copypasta',
        query_page_numbers=[3],
        tags=['pacer'],
        conjunctive=False
    )
    assert sim_search is not None, Exception("No results returned from filtered search")

    #Test conjunctive semantics
    await DB.insert(content='Text', title='Some text', page_numbers=[1], tags=['tag1'])
    await DB.insert(content='Text', title='Some mo text', page_numbers=[1], tags=['tag2', 'tag3'])
    await DB.insert(content='Text', title='Even mo text', page_numbers=[1], tags=['tag3'])

    # Using limit default
    sim_search = await DB.search(text='Text', tags=['tag1', 'tag3'], conjunctive=False)
    assert sim_search is not None, Exception("No results returned from filtered search")
    assert len(list(sim_search)) == 3, Exception(f"There should be 3 results, received {sim_search}")

    sim_search = await DB.search(text='Text', tags=['tag1', 'tag3'], conjunctive=False, limit=1000)
    assert sim_search is not None, Exception("No results returned from filtered search")
    assert len(list(sim_search)) == 3, Exception(f"There should be 3 results, received {sim_search}")

    texts = ['Hello world', 'Hello Dolly', 'Good-Bye to All That']
    authors = ['Brian Kernighan', 'Louis Armstrong', 'Robert Graves']
    metas = [[f'author={a}'] for a in authors]
    count = len(texts)
    records = zip(texts, metas, ['']*count, [None]*count)
    await DB.insert_many(records)

    sim_search = await DB.search(text='Hi there!', threshold=0.999, limit=0)
    assert sim_search is not None, Exception("No results returned from filtered search")
    assert len(list(sim_search)) == 3, Exception(f"There should be 3 results, received {sim_search}")

    sim_search = await DB.search(text='Hi there!', threshold=0.999, limit=2)
    assert sim_search is not None, Exception("No results returned from filtered search")
    assert len(list(sim_search)) == 2, Exception(f"There should be 2 results, received {sim_search}")

    await DB.drop_table()


if __name__ == '__main__':
    raise SystemExit("Attention! Run with pytest")
