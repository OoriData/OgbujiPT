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

# FIXME: Replace with a check for a running PG instance
if True:
    pytest.skip("Postgres instance/docker not available for testing PG code", allow_module_level=True)

from ogbujipt.text_helper import text_splitter

# import os
# HOST = os.environ.get('PGHOST')
# DB_NAME = 'PGv'
# PORT = 5432
# USER = 'oori'
# PASSWORD = 'example'

from unittest.mock import MagicMock, patch

from ogbujipt.embedding.pgvector import PGvectorHelper
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
    coll.db.upsert.assert_called_with(
        collection_name=collection_name,
        points=[mock_pstruct]
        )
