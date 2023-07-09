'''
pytest test

or

pytest test/test_embedding_helper.py

Uses the COME_THUNDER_POEM fixture from conftest.py
'''
# import pytest

# from qdrant_client.http.models.models import ScoredPoint

from sentence_transformers import SentenceTransformer

from ogbujipt.embedding_helper import initialize_embedding_db, upsert_embedding_db
from ogbujipt.text_helper import text_splitter


def test_embed_poem(COME_THUNDER_POEM):
    # LLM will be downloaded from HuggingFace automatically
    # FIXME: We want to mock this instead
    embedding = SentenceTransformer(DOC_EMBEDDINGS_LLM)

    # Split the chunks
    chunks = text_splitter(
        COME_THUNDER_POEM, 
        chunk_size=21, 
        chunk_overlap=3, 
        separator='\n'
        )

    collection_name = 'test_collection'

    # initialize a client
    client = initialize_embedding_db(
        chunks,
        embedding,
        collection_name
        )

    # insert the chunks into the Qdrant client
    client = upsert_embedding_db(
        client, 
        chunks, 
        embedding, 
        collection_name
        )

    embedded_question = embedding.encode(TEST_QUESTION)

    relevant_answers = client.search(
        collection_name=collection_name, 
        query_vector=embedded_question, 
        limit=1
        )

    joined_answers = ''.join(relevant_answers[0].payload['chunk_string'])

    assert 'O dancers' in joined_answers


DOC_EMBEDDINGS_LLM = 'all-MiniLM-L6-v2'

TEST_QUESTION = 'Hey dancers'
