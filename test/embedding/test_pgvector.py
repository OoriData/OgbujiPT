'''
pytest test

or

pytest test/embedding/test_pgvector.py

Uses fixtures from ../conftest.py

Useful notes for asyncio: https://tonybaloney.github.io/posts/async-test-patterns-for-pytest-and-unittest.html

Notes, if needed, for mocking PG by running a local Docker container: https://ryan-duve.medium.com/how-to-mock-postgresql-with-pytest-and-pytest-postgresql-26b4a5ea3c25

TODO: incorporate with PG/Docker in CI/CD. GitLab recipe we might be bale to adapt: https://forum.gitlab.com/t/how-to-run-pytest-with-fixtures-that-spin-up-docker-containers/57190

Another option might be: https://pypi.org/project/pytest-docker/
'''

# Remove, or narow down, once we no longer need to skip
# ruff: noqa

import pytest
from unittest.mock import MagicMock, patch

import os
from ogbujipt.embedding.pgvector import docDB
from sentence_transformers       import SentenceTransformer

# FIXME: This stanza to go away once mocking is complete - Kai
HOST = os.environ.get('PGHOST', '0.0.0.0')
DB_NAME = 'PGv'
PORT = 5432
USER = 'oori'
PASSWORD = 'example'

pacer_copypasta = [  # Demo document
    'The FitnessGram™ Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as it continues.', 
    'The 20 meter pacer test will begin in 30 seconds. Line up at the start.', 
    'The running speed starts slowly, but gets faster each minute after you hear this signal.', 
    '[beep] A single lap should be completed each time you hear this sound.', 
    '[ding] Remember to run in a straight line, and run as long as possible.', 
    'The second time you fail to complete a lap before the sound, your test is over.', 
    'The test will begin on the word start. On your mark, get ready, start.'
]

@pytest.mark.asyncio
async def test_PGv_embed_pacer():
    e_model = MagicMock(spec=SentenceTransformer)
    TABLE_NAME = 'embedding_test'
    vDB = await docDB.from_conn_params(
        embedding_model=e_model,
        table_name=TABLE_NAME,
        db_name=DB_NAME,
        host=HOST,
        port=int(PORT),
        user=USER,
        password=PASSWORD)
    
    assert vDB is not None, pytest.skip("Postgres instance/docker not available for testing PG code", allow_module_level=True)
    
    # Create tables
    await vDB.drop_table()
    await vDB.create_table()

    # Insert data
    for index, text in enumerate(pacer_copypasta):   # For each line in the copypasta
        await vDB.insert(                            # Insert the line into the table
            content=text,                            # The text to be embedded
            title=f'Pacer Copypasta line {index}',   # Title metadata
            page_numbers=[1, 2, 3],                  # Page number metadata
            tags=['fitness', 'pacer', 'copypasta'],  # Tag metadata
        )

    # search table with perfect match
    search_string = '[beep] A single lap should be completed each time you hear this sound.'
    sim_search = await vDB.search(query_string=search_string, limit=3)
    assert sim_search is not None, Exception("No results returned from perfect search")

    search_string = 'straight'
    sim_search = await vDB.search(query_string=search_string, limit=3)
    assert sim_search is not None, Exception("No results returned from straight search")

    await vDB.drop_table()

# FIXME: Fix test_PGv_embed_poem
if True:
    pytest.skip("Skipping test_PGv_embed_poem tests for now", allow_module_level=True)
from unittest.mock import MagicMock, patch

# Ugh. even this import is slow as hell, but needed if we want to spec the mocker
# time python -c "from sentence_transformers import SentenceTransformer"
# 2.78s user 4.17s system 47% cpu 14.757 total [on MacBook Pro 2020]
from sentence_transformers import SentenceTransformer

from ogbujipt import embedding_helper
from ogbujipt.text_helper import text_splitter


# @pytest.fixture
# def SENTENCE_TRANSFORMER():
#     e_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the embedding model
#     return 'And the secret thing in its heaving\nThreatens with iron mask\nThe last lighted torch of the century…'

pacer_copypasta = [  # Demo data
    "Structure of visceral layer of Bowman's capsule is a glomerular capsule structure and a structure of epithelium."
]

# FIXME: This stanza to go away once mocking is complete
import os
HOST = os.environ.get('PGHOST', 'localhost')
DB_NAME = 'PGv'
PORT = 5432
USER = 'oori'
PASSWORD = 'example'

# @patch('ogbujipt.embedding_helper.PGvectorConnection')
# @patch('sentence_transformers.SentenceTransformer')
# def test_PGv_embed_poem(mock_sentence_transformer, mock_pgvector_connection, COME_THUNDER_POEM, CORRECT_STRING):
@pytest.mark.asyncio
async def test_PGv_embed_poem(mocker):
    e_model = MagicMock(spec=SentenceTransformer)
    TABLE_NAME = 'embedding_test'
    vDB = await PGvectorHelper.from_conn_params(
        embedding_model=e_model,
        table_name=TABLE_NAME,
        db_name=DB_NAME,
        host=HOST,
        port=int(PORT),
        user=USER,
        password=PASSWORD)

    # mock_connect = MagicMock()
    # mock_cursor = MagicMock()
    # mock_cursor.fetchall.return_value = expected
    # mock_connect.cursor.return_value = mock_cursor

    # result = d.read(mock_connect)
    # self.assertEqual(result, expected)

    # TODO: Add more shape to the mocking, to increase the tests's usefulness
    embedding_helper.models = MagicMock()
    mock_vparam = object()
    embedding_helper.models.VectorParams.side_effect = [mock_vparam]

    conn = MagicMock(spec=mock_pgvector_connection).create.return_value

    # client.count.side_effect = ['count=0']
    conn.db.count.side_effect = lambda collection_name: 'count=0'
    conn.update(chunks)
    conn.db.recreate_collection.assert_called_once_with(
        collection_name='test_collection',
        vectors_config=mock_vparam
        )

    embedding_model.encode.assert_called_with(CORRECT_STRING)

    # Test update/insert into the DB
    mock_pstruct = object()
    embedding_helper.models.PointStruct.side_effect = lambda id=None, vector=None, payload=None: mock_pstruct

    conn.db.count.reset_mock()
    conn.update(chunks)

    # XXX: Add test with metadata
    conn.db.upsert.assert_called_with(
        collection_name=collection_name,
        points=[mock_pstruct]
        )
