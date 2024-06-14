# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/embedding/test_pgvector_data.py
'''
After setup as described in the README.md for this directory, run the tests with:

pytest test

or, for just this test module:

`pytest test/embedding/test_pgvector_data.py`

Uses fixtures from test/embedding/conftest.py in current & parent directories, including Postgres database connection
'''

import pytest
from unittest.mock import MagicMock, DEFAULT  # noqa: F401

import numpy as np

from ogbujipt.embedding.pgvector import match_exact, match_oneof


KG_STATEMENTS = [  # Demo data
    ("👤 Alikiba `releases` 💿 'Yalaiti'",
        {'url': 'https://njok.com/yalaiti-lyrics/', 'primary': True, 'when': '2023-11-29'}),
    ("👤 Sabah Salum `featured_in` 💿 'Yalaiti'",
        {'url': 'https://njok.com/yalaiti-lyrics/', 'primary': False, 'when': '2023-11-29'}),
    ('👤 Kukbeatz `collab_with` 👤 Ruger',
        {'url': 'https://njok.com/all-of-us-lyrics/', 'primary': True, 'when': '2023-11-25'}),
    ('💿 All of Us `song_by` 👤 Kukbeatz & Ruger',
        {'url': 'https://njok.com/all-of-us-lyrics/', 'primary': False, 'when': '2023-11-25'}),
    ('👤 Blaqbonez `collab_with` 👤 Fireboy DML',
        {'url': 'https://njok.com/fireboy-dml-collab/', 'primary': True, 'when': '2023-11-19'})
]


# XXX: Move to a fixture?
# Definitely don't want to even import SentenceTransformer class due to massive side-effects
class SentenceTransformer(object):
    def __init__(self, model_name_or_path):
        self.encode = MagicMock()


@pytest.mark.asyncio
async def test_insert_data_vector(DB):
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])

    item1_text = KG_STATEMENTS[0][0]
    item1_meta = KG_STATEMENTS[0][1]

    # Insert data
    for index, (text, meta) in enumerate(KG_STATEMENTS):
        await DB.insert(    # Insert the row into the table
            content=text,   # text to be embedded
            metadata=meta,  # Tag metadata
        )

    assert await DB.count_items() == len(KG_STATEMENTS), Exception('Incorrect number of documents after insertion')

    # search table with perfect match
    result = await DB.search(text=item1_text, limit=3)
    # assert result is not None, Exception('No results returned from perfect search')

    # Even though the embedding is mocked, the stored text should be faithful
    row = next(result)
    assert row.content == item1_text, 'text mismatch'
    assert row.metadata == item1_meta, 'Metadata mismatch'


@pytest.mark.asyncio
async def test_insertmany_data_vector(DB):
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])

    item1_text = KG_STATEMENTS[0][0]
    item1_meta = KG_STATEMENTS[0][1]

    # Insert data using insert_many()
    # dataset = ((text, metadata) for (text, metadata) in KG_STATEMENTS)
    
    await DB.insert_many(KG_STATEMENTS)

    assert await DB.count_items() == len(KG_STATEMENTS), Exception('Incorrect number of documents after insertion')

    # search table with perfect match
    result = await DB.search(text=item1_text, limit=3)
    # assert result is not None, Exception('No results returned from perfect search')

    # Even though the embedding is mocked, the stored text should be faithful
    row = next(result)
    assert row.content == item1_text
    
    # Adjusted assertion for metadata comparison
    assert row.metadata == item1_meta, "Metadata mismatch"


@pytest.mark.asyncio
async def test_search_with_filter(DB):
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])

    # item1_text = KG_STATEMENTS[0][0]

    # Insert data using insert_many()
    # dataset = ((text, metadata) for (text, metadata) in KG_STATEMENTS)

    await DB.insert_many(KG_STATEMENTS)

    # search table with perfect match, but only where primary is set to True
    primary_filt = match_exact('primary', True)
    result = list(await DB.search(text='Kukbeatz and Ruger', meta_filter=primary_filt))
    assert len(result) == 3

    result = list(await DB.search(text='Kukbeatz and Ruger', meta_filter=primary_filt, limit=1))
    assert len(result) == 1


@pytest.mark.asyncio
async def test_search_with_date_filter(DB):
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])

    # item1_text = KG_STATEMENTS[0][0]

    # Insert data using insert_many()
    # dataset = ((text, metadata) for (text, metadata) in KG_STATEMENTS)

    await DB.insert_many(KG_STATEMENTS)

    # search table with perfect match, but only where primary is set to True
    primary_filt = match_exact('when', '2023-11-29')
    result = list(await DB.search(text='Kukbeatz and Ruger', meta_filter=primary_filt))
    assert len(result) == 2

    primary_filt = match_exact('when', '2023-11-19')
    result = list(await DB.search(text='Kukbeatz and Ruger', meta_filter=primary_filt))
    assert len(result) == 1


@pytest.mark.asyncio
async def test_search_with_date_filter_match_oneof(DB):
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])

    # item1_text = KG_STATEMENTS[0][0]

    # Insert data using insert_many()
    # dataset = ((text, metadata) for (text, metadata) in KG_STATEMENTS)

    await DB.insert_many(KG_STATEMENTS)

    # search table with perfect match, but only where primary is set to True
    primary_filt = match_oneof('when', ('2023-11-29',))
    result = list(await DB.search(text='Kukbeatz and Ruger', meta_filter=primary_filt))
    assert len(result) == 2

    primary_filt = match_oneof('when', ('2023-11-29', '2023-11-19'))
    result = list(await DB.search(text='Kukbeatz and Ruger', meta_filter=primary_filt))
    assert len(result) == 3

    primary_filt = match_oneof('when', ('2023-11-29', '2023-11-25', '2023-11-19'))
    result = list(await DB.search(text='Kukbeatz and Ruger', meta_filter=primary_filt))
    assert len(result) == 5


if __name__ == '__main__':
    raise SystemExit("Attention! Run with pytest")
