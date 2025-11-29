# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/embedding/test_pgvector_message.py
'''
After setup as described in the README.md for this directory, run the tests with:

pytest test

or, for just this test module:

pytest test/embedding/test_pgvector_message.py

Uses fixtures from test/embedding/conftest.py in current & parent directories, including Postgres database connection
'''

from datetime import datetime

import pytest
from unittest.mock import MagicMock, DEFAULT  # noqa: F401

import numpy as np

# XXX: Move to a fixture?
# Definitely don't want to even import SentenceTransformer class due to massive side-effects
class SentenceTransformer(object):
    def __init__(self, model_name_or_path):
        self.encode = MagicMock()


@pytest.fixture
def MESSAGES():
    messages = [  # Test data: history_key, role, content, timestamp, metadata
    ('00000000-0000-0000-0000-000000000000', 'ama', 'Hello Eme!', '2021-10-01 00:00:00+00:00', {'1': 'a'}),
    ('00000000-0000-0000-0000-000000000001', 'ugo', 'Greetings Ego', '2021-10-01 00:00:01+00:00', {'2': 'b'}),
    ('00000000-0000-0000-0000-000000000000', 'eme', 'How you dey, Ama!', '2021-10-01 00:00:02+00:00', {'3': 'c'}),
    ('00000000-0000-0000-0000-000000000001', 'ego', 'What a pleasant surprise', '2021-10-01 00:00:03+00:00', {'4': 'd'}),  # noqa: E501
    ('00000000-0000-0000-0000-000000000000', 'ama', 'Not bad, not bad at all', '2021-10-01 00:00:04+00:00', {'5': 'e'}),  # noqa: E501
    ('00000000-0000-0000-0000-000000000001', 'ugo', 'Glad you think so. I was planning to drop by later', '2021-10-01 00:00:05+00:00', {'6': 'f'}),  # noqa: E501
    ('00000000-0000-0000-0000-000000000000', 'eme', 'Very good. Say hello to your family for me.', '2021-10-01 00:00:06+00:00', {'7': 'g'}),  # noqa: E501
    ('00000000-0000-0000-0000-000000000001', 'ugo', 'An even better surprise, I hope!', '2021-10-01 00:00:07+00:00', {'8': 'h'})  # noqa: E501
    ]
    return [(history_key, role, content, datetime.fromisoformat(timestamp), metadata)
    for history_key, role, content, timestamp, metadata in messages
    ]


@pytest.mark.asyncio
async def test_insert_message_vector(DB, MESSAGES):
    # Insert data
    for index, (row) in enumerate(MESSAGES):
        await DB.insert(*row)

    assert await DB.count_items() == len(MESSAGES), Exception('Incorrect number of messages after insertion')

    history_key, role, content, timestamp, metadata = MESSAGES[0]

    # search table with perfect match
    result = await DB.search(text=content, history_key=history_key, limit=3)
    # assert result is not None, Exception('No results returned from perfect search')

    # Even though the embedding is mocked, the stored text should be faithful
    row = next(result)
    assert row.content == content
    assert row.metadata == metadata


@pytest.mark.asyncio
async def test_insert_message_vector_windowed(DB_WINDOWED2, MESSAGES):
    assert await DB_WINDOWED2.count_items() == 0, Exception('Starting with incorrect number of messages')
    # Insert data
    for index, (row) in enumerate(MESSAGES):
        await DB_WINDOWED2.insert(*row)

    # There should be 2 left from each history key
    assert await DB_WINDOWED2.count_items() == 4, Exception('Incorrect number of messages after insertion')

    # In the windowed case, the oldest 4 messages should have been deleted
    history_key, role, content, timestamp, metadata = MESSAGES[5]

    # search table with perfect match
    result = await DB_WINDOWED2.search(text=content, history_key=history_key, limit=2)
    # assert result is not None, Exception('No results returned from perfect search')

    # Even though the embedding is mocked, the stored text should be faithful
    row = next(result)
    assert row.content == content
    assert row.metadata == metadata


@pytest.mark.asyncio
async def test_insertmany_message_vector(DB, MESSAGES):
    # Insert data using insert_many()
    await DB.insert_many(MESSAGES)

    assert await DB.count_items() == len(MESSAGES), Exception('Incorrect number of messages after insertion')

    history_key, role, content, timestamp, metadata = MESSAGES[0]

    # search table with perfect match
    result = await DB.search(text=content, history_key=history_key, limit=3)
    # assert result is not None, Exception('No results returned from perfect search')

    # Even though the embedding is mocked, the stored text should be faithful
    row = next(result)
    assert row.content == content
    assert row.metadata == metadata


@pytest.mark.asyncio
async def test_insertmany_message_vector_windowed(DB_WINDOWED2, MESSAGES):
    assert await DB_WINDOWED2.count_items() == 0, Exception('Starting with incorrect number of messages')
    # Insert data using insert_many()
    await DB_WINDOWED2.insert_many(MESSAGES)

    # There should be 2 left from each history key
    assert await DB_WINDOWED2.count_items() == 4, Exception('Incorrect number of messages after insertion')

    # In the windowed case, the oldest 4 messages should have been deleted
    history_key, role, content, timestamp, metadata = MESSAGES[5]

    # search table with perfect match
    result = await DB_WINDOWED2.search(text=content, history_key=history_key, limit=3)
    # assert result is not None, Exception('No results returned from perfect search')

    # Even though the embedding is mocked, the stored text should be faithful
    row = next(result)
    assert row.content == content
    assert row.metadata == metadata


@pytest.mark.asyncio
async def test_get_messages_all_limit(DB, MESSAGES):
    # Insert data using insert_many()
    await DB.insert_many(MESSAGES)

    history_key, role, content, timestamp, metadata = MESSAGES[0]

    results = await DB.get_messages(history_key=history_key)
    assert len(list(results)) == 4, Exception('Incorrect number of messages returned from chatlog')

    results = await DB.get_messages(history_key=history_key, limit=3)
    assert len(list(results)) == 3, Exception('Incorrect number of messages returned from chatlog')

    # With limit, should return the most recent messages
    history_key, role, content, timestamp, metadata = MESSAGES[-1]

    results = list(await DB.get_messages(history_key=history_key, limit=1))
    assert len(results) == 1, Exception('Incorrect number of messages returned from chatlog')
    assert results[0].content == content, Exception('Incorrect message returned from chatlog')


@pytest.mark.asyncio
async def test_get_messages_since(DB, MESSAGES):
    await DB.insert_many(MESSAGES)

    history_key, role, content, timestamp, metadata = MESSAGES[0]

    since_ts = datetime.fromisoformat('2021-10-01 00:00:03+00:00')
    results = list(await DB.get_messages(history_key=history_key, since=since_ts))
    assert len(results) == 2, Exception('Incorrect number of messages returned from chatlog')

    since_ts = datetime.fromisoformat('2021-10-01 00:00:04+00:00')
    results = list(await DB.get_messages(history_key=history_key, since=since_ts))
    assert len(results) == 1, Exception('Incorrect number of messages returned from chatlog')


@pytest.mark.asyncio
async def test_search_threshold(DB, MESSAGES):
    dummy_model = SentenceTransformer('mock_transformer')
    def encode_tweaker(*args, **kwargs):
        # Note: cosine similarity of [1, 2, 3] & [100, 300, 500] appears to be ~ 0.9939
        if args[0].startswith('Hi'):
            return np.array([100, 300, 500])
        else:
            return np.array([1, 2, 3])

    dummy_model.encode.side_effect = encode_tweaker
    # Need to replace the default encoder set up by the fixture
    DB._embedding_model = dummy_model

    await DB.insert_many(MESSAGES)

    history_key, role, content, timestamp, metadata = MESSAGES[0]

    results = list(await DB.search(history_key, 'Hi!', threshold=0.999))
    assert results is not None and len(results) == 0

    results = list(await DB.search(history_key, 'Hi!', threshold=0.5))
    assert results is not None and len(results) == 1


if __name__ == '__main__':
    raise SystemExit("Attention! Run with pytest")
