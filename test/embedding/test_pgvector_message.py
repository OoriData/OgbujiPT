# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/embedding/test_pgvector_message.py
'''
See test/embedding/test_pgvector.py for important notes on running these tests
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
    assert row.metadata == {'1': 'a'}


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
    assert row.metadata == {'1': 'a'}


@pytest.mark.asyncio
async def test_get_chatlog_all_limit(DB, MESSAGES):
    # Insert data using insert_many()
    await DB.insert_many(MESSAGES)

    history_key, role, content, timestamp, metadata = MESSAGES[0]

    results = await DB.get_chatlog(history_key=history_key)
    assert len(list(results)) == 4, Exception('Incorrect number of messages returned from chatlog')

    results = await DB.get_chatlog(history_key=history_key, limit=3)
    assert len(list(results)) == 3, Exception('Incorrect number of messages returned from chatlog')


@pytest.mark.asyncio
async def test_get_chatlog_since(DB, MESSAGES):
    await DB.insert_many(MESSAGES)

    history_key, role, content, timestamp, metadata = MESSAGES[0]

    since_ts = datetime.fromisoformat('2021-10-01 00:00:03+00:00')
    results = list(await DB.get_chatlog(history_key=history_key, since=since_ts))
    assert len(results) == 2, Exception('Incorrect number of messages returned from chatlog')

    since_ts = datetime.fromisoformat('2021-10-01 00:00:04+00:00')
    results = list(await DB.get_chatlog(history_key=history_key, since=since_ts))
    assert len(results) == 1, Exception('Incorrect number of messages returned from chatlog')

if __name__ == '__main__':
    raise SystemExit("Attention! Run with pytest")
