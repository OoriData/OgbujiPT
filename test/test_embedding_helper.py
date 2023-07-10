'''
pytest test

or

pytest test/test_embedding_helper.py

Uses the COME_THUNDER_POEM fixture from conftest.py
'''
import pytest

from ogbujipt import embedding_helper
from ogbujipt.embedding_helper import qdrant_init_embedding_db, \
    qdrant_add_collection, qdrant_upsert_collection
from ogbujipt.text_helper import text_splitter


@pytest.fixture
def CORRECT_STRING():
    return 'And the secret thing in its heaving\nThreatens with iron mask\nThe last lighted torch of the century…'


def test_embed_poem(mocker, COME_THUNDER_POEM, CORRECT_STRING):
    # LLM will be downloaded from HuggingFace automatically
    # FIXME: We want to mock this instead
    # Split the chunks
    chunks = text_splitter(
        COME_THUNDER_POEM, 
        chunk_size=21, 
        chunk_overlap=3, 
        separator='\n'
        )

    collection_name = 'test_collection'

    # TODO: Add more shape to the mocking, to increase the tests's usefulness
    embedding = mocker.MagicMock()
    embedding_helper.models = mocker.MagicMock()
    mock_vparam = object()
    embedding_helper.models.VectorParams.side_effect = [mock_vparam]
    mocker.patch('ogbujipt.embedding_helper.QdrantClient')

    client = qdrant_init_embedding_db()

    #client.count.side_effect = ['count=0']
    client.count.side_effect = lambda collection_name: 'count=0'
    client = qdrant_add_collection(
        client,
        chunks,
        embedding,
        collection_name
        )
    client.recreate_collection.assert_called_once_with(
        collection_name='test_collection',
        vectors_config=mock_vparam
        )
    
    embedding.encode.assert_called_with(CORRECT_STRING)

    # Test update/insert into the DB
    mock_pstruct = object()
    embedding_helper.models.PointStruct.side_effect = lambda id=None, vector=None, payload=None: mock_pstruct
    
    client.count.reset_mock()
    client = qdrant_upsert_collection(
        client, 
        chunks, 
        embedding, 
        collection_name
        )

    client.upsert.assert_called_with(
        collection_name=collection_name,
        points=[mock_pstruct]
        )
