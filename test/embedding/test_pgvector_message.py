# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/embedding/test_pgvector_message.py
'''
See test/embedding/test_pgvector.py for important notes on running these tests
'''

import os
from datetime import datetime

import pytest
from unittest.mock import MagicMock, DEFAULT  # noqa: F401

import numpy as np

from ogbujipt.embedding.pgvector import MessageDB

# XXX: This stanza to go away once mocking is complete - Kai
HOST = os.environ.get('PG_HOST', 'localhost')
DB_NAME = os.environ.get('PG_DATABASE', 'mock_db')
USER = os.environ.get('PG_USER', 'mock_user')
PASSWORD = os.environ.get('PG_PASSWORD', 'mock_password')
PORT = os.environ.get('PG_PORT', 5432)


MESSAGES = [  # Test data: history_key, role, content, timestamp, metadata
    ('00000000-0000-0000-0000-000000000000', 'ama', 'Hello Eme!', '2021-10-01 00:00:00+00:00', {'1': 'a'}),
    ('00000000-0000-0000-0000-000000000001', 'ugo', 'Greetings Ego', '2021-10-01 00:00:01+00:00', {'2': 'b'}),
    ('00000000-0000-0000-0000-000000000000', 'eme', 'How you dey, Ama!', '2021-10-01 00:00:02+00:00', {'3': 'c'}),
    ('00000000-0000-0000-0000-000000000001', 'ego', 'What a pleasant surprise', '2021-10-01 00:00:03+00:00', {'4': 'd'}),  # noqa: E501
    ('00000000-0000-0000-0000-000000000000', 'ama', 'Not bad, not bad at all', '2021-10-01 00:00:04+00:00', {'5': 'e'}),  # noqa: E501
    ('00000000-0000-0000-0000-000000000001', 'ugo', 'Glad you think so. I was planning to drop by later', '2021-10-01 00:00:05+00:00', {'6': 'f'}),  # noqa: E501
    ('00000000-0000-0000-0000-000000000000', 'eme', 'Very good. Say hello to your family for me.', '2021-10-01 00:00:06+00:00', {'7': 'g'}),  # noqa: E501
    ('00000000-0000-0000-0000-000000000001', 'ugo', 'An even better surprise, I hope!', '2021-10-01 00:00:07+00:00', {'8': 'h'})  # noqa: E501
    ]
MESSAGES = [
    (history_key, role, content, datetime.fromisoformat(timestamp), metadata)
    for history_key, role, content, timestamp, metadata in MESSAGES
]


# XXX: Move to a fixture?
# Definitely don't want to even import SentenceTransformer class due to massive side-effects
class SentenceTransformer(object):
    def __init__(self, model_name_or_path):
        self.encode = MagicMock()


@pytest.mark.asyncio
async def test_insert_message_vector():
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

    # Insert data
    for index, (row) in enumerate(MESSAGES):
        await vDB.insert(*row)

    assert await vDB.count_items() == len(MESSAGES), Exception('Incorrect number of messages after insertion')

    history_key, role, content, timestamp, metadata = MESSAGES[0]

    # search table with perfect match
    result = await vDB.search(text=content, history_key=history_key, limit=3)
    # assert result is not None, Exception('No results returned from perfect search')

    # Even though the embedding is mocked, the stored text should be faithful
    row = next(result)
    assert row.content == content
    assert row.metadata == {'1': 'a'}

    await vDB.drop_table()


@pytest.mark.asyncio
async def test_insertmany_message_vector():
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

    # Insert data using insert_many()
    await vDB.insert_many(MESSAGES)

    assert await vDB.count_items() == len(MESSAGES), Exception('Incorrect number of messages after insertion')

    history_key, role, content, timestamp, metadata = MESSAGES[0]

    # search table with perfect match
    result = await vDB.search(text=content, history_key=history_key, limit=3)
    # assert result is not None, Exception('No results returned from perfect search')

    # Even though the embedding is mocked, the stored text should be faithful
    row = next(result)
    assert row.content == content
    assert row.metadata == {'1': 'a'}

    await vDB.drop_table()


if __name__ == '__main__':
    raise SystemExit("Attention! Run with pytest")
