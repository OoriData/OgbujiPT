# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/embedding/test_pgvector_data.py
'''
After setup as described in the README.md for this directory, run the tests with:

pytest test

or, for just this test module:

`pytest test/embedding/test_pgvector_data.py`

Uses fixtures from conftest.py in current & parent directories
'''

import pytest
from unittest.mock import MagicMock, DEFAULT  # noqa: F401

import numpy as np


KG_STATEMENTS = [  # Demo data
    ("👤 Alikiba `releases_single` 💿 'Yalaiti'", {'url': 'https://notjustok.com/lyrics/yalaiti-lyrics-by-alikiba-ft-sabah-salum/'}),
    ("👤 Sabah Salum `featured_in` 💿 'Yalaiti'", {'url': 'https://notjustok.com/lyrics/yalaiti-lyrics-by-alikiba-ft-sabah-salum/'}),
    ('👤 Kukbeatz `collaborates_with` 👤 Ruger', {'url': 'https://notjustok.com/lyrics/all-of-us-lyrics-by-kukbeatz-ft-ruger/'}),
    ('💿 All of Us `is_a_song_by` 👤 Kukbeatz and Ruger', {'url': 'https://notjustok.com/lyrics/all-of-us-lyrics-by-kukbeatz-ft-ruger/'}),
    ('👤 Blaqbonez `collaborates_with` 👤 Fireboy DML', {'url': 'https://notjustok.com/news/snippet-of-fireboy-dmls-collaboration-with-blaqbonez/'})
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
        await DB.insert(                                   # Insert the row into the table
            content=text,                                  # text to be embedded
            tags=[f'{k}={v}' for (k, v) in meta.items()],  # Tag metadata
        )

    assert await DB.count_items() == len(KG_STATEMENTS), Exception('Incorrect number of documents after insertion')

    # search table with perfect match
    result = await DB.search(text=item1_text, limit=3)
    # assert result is not None, Exception('No results returned from perfect search')

    # Even though the embedding is mocked, the stored text should be faithful
    row = next(result)
    assert row.content == item1_text
    assert row.tags == [f'{k}={v}' for (k, v) in item1_meta.items()]

    await DB.drop_table()


@pytest.mark.asyncio
async def test_insertmany_data_vector(DB):
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])

    item1_text = KG_STATEMENTS[0][0]
    item1_meta = KG_STATEMENTS[0][1]

    # Insert data using insert_many()
    dataset = ((text, [f'{k}={v}' for (k, v) in tags.items()]) for (text, tags) in KG_STATEMENTS)
    
    await DB.insert_many(dataset)

    assert await DB.count_items() == len(KG_STATEMENTS), Exception('Incorrect number of documents after insertion')

    # search table with perfect match
    result = await DB.search(text=item1_text, limit=3)
    # assert result is not None, Exception('No results returned from perfect search')

    # Even though the embedding is mocked, the stored text should be faithful
    row = next(result)
    assert row.content == item1_text
    assert row.tags == [f'{k}={v}' for (k, v) in item1_meta.items()]

    await DB.drop_table()


if __name__ == '__main__':
    raise SystemExit("Attention! Run with pytest")
