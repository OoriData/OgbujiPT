# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding_helper

'''
Routines to help with embedding for vector databases such as Qdrant
'''
from qdrant_client import QdrantClient
from qdrant_client.http import models


def initialize_embedding_db(chunks, embedding, collection_name, distance_func='Cosine'):
    '''

    '''
    partial_embeddings = embedding.encode(chunks[:1])
    vector_size = len(partial_embeddings[0])

    distance_func = distance_func.lower().capitalize()

    client = QdrantClient(':memory:')

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size, 
            distance=distance_func
            )
        )

    return client


def upsert_chunks(client, chunks, embedding, collection_name):
    '''

    '''
    current_count = int(str(client.count(collection_name)).partition('=')[-1])

    for id, chunk in enumerate(chunks):
        vectorized_chunk = list(embedding.encode(chunk))

        prepped_chunk = [float(vector) for vector in vectorized_chunk]

        prepped_payload = {'chunk_text': chunk}

        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=id + current_count,
                    vector=prepped_chunk,
                    payload=prepped_payload
                    )
                ]
            )

    return client
