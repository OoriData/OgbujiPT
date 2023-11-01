'''
pytest test

or

pytest test/embedding/test_pgvector.py

Uses the COME_THUNDER_POEM fixture from conftest.py
'''
import pytest

from ogbujipt.text_helper import text_splitter

# import os
# HOST = os.environ.get('PGHOST')
# DB_NAME = 'PGv'
# PORT = 5432
# USER = 'oori'
# PASSWORD = 'example'

from ogbujipt.embedding.pgvector import PGvectorHelper
from sentence_transformers     import SentenceTransformer

@pytest.fixture
def SENTENCE_TRANSFORMER():
    e_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the embedding model
    return 'And the secret thing in its heaving\nThreatens with iron mask\nThe last lighted torch of the century…'

from unittest.mock import MagicMock, patch

from ogbujipt import embedding_helper
from ogbujipt.text_helper import text_splitter

pacer_copypasta = [  # Demo data
    "Structure of visceral layer of Bowman's capsule is a glomerular capsule structure and a structure of epithelium."
]

@patch('ogbujipt.embedding_helper.PGvectorConnection')
@patch('sentence_transformers.SentenceTransformer')
def test_PGv_embed_poem(mock_sentence_transformer, mock_pgvector_connection, COME_THUNDER_POEM, CORRECT_STRING):
    # LLM will be downloaded from HuggingFace automatically
    # FIXME: We want to mock this instead, or rather just have a fixture with the results
    # Split the chunks
    chunks = text_splitter(
        COME_THUNDER_POEM, 
        chunk_size=21, 
        chunk_overlap=3, 
        separator='\n'
        )

    # TODO: Add more shape to the mocking, to increase the tests's usefulness
    embedding_model = MagicMock(spec=mock_sentence_transformer)
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
