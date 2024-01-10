# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/embedding/test_pgvector_message.py
'''
See test/embedding/test_pgvector.py for important notes on running these tests
'''

import pytest
from unittest.mock import MagicMock, DEFAULT  # noqa: F401

import os
from ogbujipt.embedding.pgvector import MessageDB
import numpy as np

# XXX: This stanza to go away once mocking is complete - Kai
HOST = os.environ.get('PG_HOST', 'localhost')
DB_NAME = os.environ.get('PG_DATABASE', 'mock_db')
USER = os.environ.get('PG_USER', 'mock_user')
PASSWORD = os.environ.get('PG_PASSWORD', 'mock_password')
PORT = os.environ.get('PG_PORT', 5432)

MESSAGES = [  # Test data: timetsamp, history_key, role, content, metadata
    ('2021-10-01 00:00:00+00:00', '00000000-0000-0000-0000-000000000000', 'ama', 'Hello Eme!', {'1': 'a'}),
    ('2021-10-01 00:00:01+00:00', '00000000-0000-0000-0000-000000000001', 'ugo', 'Greetings Ego', {'2': 'b'}),
    ('2021-10-01 00:00:02+00:00', '00000000-0000-0000-0000-000000000000', 'eme', 'How you dey, Ama!', {'3': 'c'}),
    ('2021-10-01 00:00:03+00:00', '00000000-0000-0000-0000-000000000001', 'ego', 'What a pleasant surprise', {'4': 'd'}),
    ('2021-10-01 00:00:04+00:00', '00000000-0000-0000-0000-000000000000', 'ama', 'Not bad, not bad at all', {'5': 'e'}),
    ('2021-10-01 00:00:05+00:00', '00000000-0000-0000-0000-000000000001', 'ugo', 'Glad you think so. I was planning to drop by later', {'6': 'f'}),  # noqa: E501
    ('2021-10-01 00:00:06+00:00', '00000000-0000-0000-0000-000000000000', 'eme', 'Very good. Say hello to your family for me.', {'7': 'g'}),  # noqa: E501
    ('2021-10-01 00:00:07+00:00', '00000000-0000-0000-0000-000000000001', 'ugo', 'An even better surprise, I hope!', {'8': 'h'})  # noqa: E501
    ]


# XXX: Move to a fixture?
# Definitely don't want to even import SentenceTransformer class due to massive side-effects
class SentenceTransformer(object):
    def __init__(self, model_name_or_path):
        self.encode = MagicMock()


@pytest.mark.asyncio
async def test_insert_data_vector():
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])
    TABLE_NAME = 'embedding_msg_test'
    try:
        vDB = await MessageDB.from_conn_params(
            embedding_model=dummy_model,
            table_name=TABLE_NAME,
            db_name=DB_NAME,
            host=HOST,
            port=int(PORT),
            user=USER,
            password=PASSWORD)
    except ConnectionRefusedError:
        pytest.skip("No Postgres instance made available for test. Skipping.", allow_module_level=True)
    if vDB is None:
        pytest.skip("No Postgres instance made available for test. Skipping.", allow_module_level=True)

    # Create tables
    await vDB.drop_table()
    assert await vDB.table_exists() is False, Exception("Table exists before creation")
    await vDB.create_table()
    assert await vDB.table_exists() is True, Exception("Table does not exist after creation")

    item1_text = KG_STATEMENTS[0][0]
    item1_meta = KG_STATEMENTS[0][1]

    # Insert data
    for index, (text, meta) in enumerate(KG_STATEMENTS):
        await vDB.insert(                                  # Insert the row into the table
            content=text,                                  # text to be embedded
            tags=[f'{k}={v}' for (k, v) in meta.items()],  # Tag metadata
        )

    assert await vDB.count_items() == len(KG_STATEMENTS), Exception('Incorrect number of documents after insertion')

    # search table with perfect match
    result = await vDB.search(text=item1_text, limit=3)
    # assert result is not None, Exception('No results returned from perfect search')

    # Even though the embedding is mocked, the stored text should be faithful
    row = next(result)
    assert row.content == item1_text
    assert row.tags == [f'{k}={v}' for (k, v) in item1_meta.items()]

    await vDB.drop_table()


@pytest.mark.asyncio
async def test_insertmany_data_vector():
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])
    # print(f'EMODEL: {dummy_model}')
    TABLE_NAME = 'embedding_test'
    try:
        vDB = await MessageDB.from_conn_params(
            embedding_model=dummy_model,
            table_name=TABLE_NAME,
            db_name=DB_NAME,
            host=HOST,
            port=int(PORT),
            user=USER,
            password=PASSWORD)
    except ConnectionRefusedError:
        pytest.skip("No Postgres instance made available for test. Skipping.", allow_module_level=True)
    if vDB is None:
        pytest.skip("No Postgres instance made available for test. Skipping.", allow_module_level=True)

    item1_text = KG_STATEMENTS[0][0]
    item1_meta = KG_STATEMENTS[0][1]

    # Create tables
    await vDB.drop_table()
    assert await vDB.table_exists() is False, Exception("Table exists before creation")
    await vDB.create_table()
    assert await vDB.table_exists() is True, Exception("Table does not exist after creation")

    # Insert data using insert_many()
    dataset = ((text, [f'{k}={v}' for (k, v) in tags.items()]) for (text, tags) in KG_STATEMENTS)
    
    await vDB.insert_many(dataset)

    assert await vDB.count_items() == len(KG_STATEMENTS), Exception('Incorrect number of documents after insertion')

    # search table with perfect match
    result = await vDB.search(text=item1_text, limit=3)
    # assert result is not None, Exception('No results returned from perfect search')

    # Even though the embedding is mocked, the stored text should be faithful
    row = next(result)
    assert row.content == item1_text
    assert row.tags == [f'{k}={v}' for (k, v) in item1_meta.items()]

    await vDB.drop_table()

