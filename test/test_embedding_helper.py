'''
pytest test

or

pytest test/test_embedding_helper.py

Uses the COME_THUNDER_POEM fixture from conftest.py
'''
import pytest

from ogbujipt import embedding_helper
from ogbujipt.embedding_helper import qdrant_collection
# from ogbujipt.embedding_helper import PGvectorConnection
from ogbujipt.text_helper import text_splitter

embedding_helper.QDRANT_AVAILABLE = True


@pytest.fixture
def CORRECT_STRING():
    return 'And the secret thing in its heaving\nThreatens with iron mask\nThe last lighted torch of the century…'


class SentenceTransformer:
    """
    Fake class for testing
    """
    def __init__(self, model_name_or_path = None,
                 modules= None,
                 device = None,
                 cache_folder = None,
                 use_auth_token = None
                 ):
        pass


    def encode(self, sentences,
               batch_size: int = 32,
               show_progress_bar= None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device= None,
               normalize_embeddings: bool = False):
        pass


def test_qdrant_embed_poem(mocker, COME_THUNDER_POEM, CORRECT_STRING):
    # LLM will be downloaded from HuggingFace automatically
    # FIXME: We want to mock this instead, or rather just have a fixture with the results
    # Split the chunks
    chunks = text_splitter(
        COME_THUNDER_POEM, 
        chunk_size=21, 
        chunk_overlap=3, 
        separator='\n'
        )

    collection_name = 'test_collection'

    # TODO: Add more shape to the mocking, to increase the tests's usefulness
    embedding_model = mocker.MagicMock(spec=SentenceTransformer)
    embedding_helper.models = mocker.MagicMock()
    mock_vparam = object()
    embedding_helper.models.VectorParams.side_effect = [mock_vparam]
    mocker.patch('ogbujipt.embedding_helper.QdrantClient')

    coll = qdrant_collection(name=collection_name, embedding_model=embedding_model)

    # client.count.side_effect = ['count=0']
    coll.db.count.side_effect = lambda collection_name: 'count=0'
    coll.update(chunks)
    coll.db.recreate_collection.assert_called_once_with(
        collection_name='test_collection',
        vectors_config=mock_vparam
        )

    embedding_model.encode.assert_called_with(CORRECT_STRING)

    # Test update/insert into the DB
    mock_pstruct = object()
    embedding_helper.models.PointStruct.side_effect = lambda id=None, vector=None, payload=None: mock_pstruct

    coll.db.count.reset_mock()
    coll.update(chunks)

    # XXX: Add test with metadata
    coll.db.upsert.assert_called_with(
        collection_name=collection_name,
        points=[mock_pstruct]
        )

# def test_PGv_embed_poem(mocker, COME_THUNDER_POEM, CORRECT_STRING):
#     # LLM will be downloaded from HuggingFace automatically
#     # FIXME: We want to mock this instead, or rather just have a fixture with the results
#     # Split the chunks
#     chunks = text_splitter(
#         COME_THUNDER_POEM, 
#         chunk_size=21, 
#         chunk_overlap=3, 
#         separator='\n'
#         )

#     # TODO: Add more shape to the mocking, to increase the tests's usefulness
#     embedding_model = mocker.MagicMock(spec=SentenceTransformer)
#     embedding_helper.models = mocker.MagicMock()
#     mock_vparam = object()
#     embedding_helper.models.VectorParams.side_effect = [mock_vparam]
#     mocker.patch('ogbujipt.embedding_helper.PGvectorConnection')

#     conn = PGvectorConnection.create(
#         embedding_model=embedding_model,
#         user='oori',
#         password='example',
#         db_name='PGv',
#         host='sofola',
#         port=5432
#         )


#     # client.count.side_effect = ['count=0']
#     conn.db.count.side_effect = lambda collection_name: 'count=0'
#     conn.update(chunks)
#     conn.db.recreate_collection.assert_called_once_with(
#         collection_name='test_collection',
#         vectors_config=mock_vparam
#         )

#     embedding_model.encode.assert_called_with(CORRECT_STRING)

#     # Test update/insert into the DB
#     mock_pstruct = object()
#     embedding_helper.models.PointStruct.side_effect = lambda id=None, vector=None, payload=None: mock_pstruct

#     conn.db.count.reset_mock()
#     conn.update(chunks)

#     # XXX: Add test with metadata
#     coll.db.upsert.assert_called_with(
#         collection_name=collection_name,
#         points=[mock_pstruct]
#         )