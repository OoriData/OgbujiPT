'''
pytest test

or

pytest test/test_embedding_helper.py
'''
import pytest

from ogbujipt.text_helper import text_splitter

from sentence_transformers import SentenceTransformer

from ogbujipt.embedding_helper import initialize_embedding_db, upsert_embedding_db

from qdrant_client.http.models.models import ScoredPoint


def test_embed_poem():
    # LLM will be downloaded from HuggingFace automatically
    embedding = SentenceTransformer(DOC_EMBEDDINGS_LLM)

    # Split the chunks
    chunks = text_splitter(
        COME_THUNDER, 
        chunk_size=21, 
        chunk_overlap=3, 
        separator='\n'
        )

    collection_name = 'test_collection'

    # initialize a client
    client = initialize_embedding_db(
        collection_name=collection_name, 
        chunks=chunks, 
        embedding=embedding
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

# One of Christopher Okigbo's greatest poems
COME_THUNDER = '''\
Now that the triumphant march has entered the last street corners,
Remember, O dancers, the thunder among the clouds…

Now that the laughter, broken in two, hangs tremulous between the teeth,
Remember, O Dancers, the lightning beyond the earth…

The smell of blood already floats in the lavender-mist of the afternoon.
The death sentence lies in ambush along the corridors of power;
And a great fearful thing already tugs at the cables of the open air,
A nebula immense and immeasurable, a night of deep waters —
An iron dream unnamed and unprintable, a path of stone.

The drowsy heads of the pods in barren farmlands witness it,
The homesteads abandoned in this century’s brush fire witness it:
The myriad eyes of deserted corn cobs in burning barns witness it:
Magic birds with the miracle of lightning flash on their feathers…

The arrows of God tremble at the gates of light,
The drums of curfew pander to a dance of death;

And the secret thing in its heaving
Threatens with iron mask
The last lighted torch of the century…'''


BASIC_EVEN_BLOCK = '''\
0123456789
abcdefghij
ABCDEFGHIJ
klmnopqrst
'''
