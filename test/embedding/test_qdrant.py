# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/embedding/test_qdrant.py
'''
pytest test

or

pytest test/embedding/test_qdrant.py

Uses the COME_THUNDER_POEM fixture from conftest.py
'''
# import pytest

from ogbujipt.embedding import qdrant
from ogbujipt.embedding.qdrant import collection
from ogbujipt.text_helper import text_split_fuzzy

qdrant.QDRANT_AVAILABLE = True


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
    chunks = text_split_fuzzy(
        COME_THUNDER_POEM, 
        chunk_size=21, 
        chunk_overlap=3, 
        separator='\n'
        )

    collection_name = 'test_collection'

    # TODO: Add more shape to the mocking, to increase the tests's usefulness
    embedding_model = mocker.MagicMock(spec=SentenceTransformer)
    qdrant.models = mocker.MagicMock()
    mock_vparam = object()
    qdrant.models.VectorParams.side_effect = [mock_vparam]
    mocker.patch('ogbujipt.embedding.qdrant.QdrantClient')

    coll = collection(name=collection_name, embedding_model=embedding_model)

    # client.count.side_effect = ['count=0']
    coll.db.count.side_effect = lambda collection_name: 'count=0'
    # FIXME: Remove list() once #85 is fixed
    coll.update(list(chunks))
    coll.db.recreate_collection.assert_called_once_with(
        collection_name='test_collection',
        vectors_config=mock_vparam
        )

    embedding_model.encode.assert_called_with(CORRECT_STRING)

    # Test update/insert into the DB
    mock_pstruct = qdrant.models.PointStruct()
    qdrant.models.PointStruct.side_effect = lambda id=None, vector=None, payload=None: mock_pstruct

    coll.db.count.reset_mock()
    coll.update(list(chunks))

    # XXX: Add test with metadata
    coll.db.upsert.assert_called_with(
        collection_name=collection_name,
        points=[mock_pstruct]
        )


if __name__ == '__main__':
    raise SystemExit("Attention! Run with pytest")
