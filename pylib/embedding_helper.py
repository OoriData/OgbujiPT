# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding_helper

'''
Routines to help with embedding for vector databases such as Qdrant
'''

from qdrant_client import QdrantClient
from qdrant_client.http import models


def initialize_embedding_db(
        chunks, 
        embedding_model, 
        collection_name, 
        distance_function='Cosine'
        ) -> QdrantClient:
    '''
    Set up a Qdrant client and a collection of embeddings inside of it.

    Args:
        chunks (List[str]): List of similar length strings to vectorize

        embedding (SentenceTransformer): SentenceTransformer object of your choice
        SentenceTransformer](https://huggingface.co/sentence-transformers)

        collection_name (str): Name that describes "chunks"

        distance_function (str): Distance function by which vectors will be compared

    Returns:
        client (QdrantClient): Initialized Qdrant client object
    '''
    # Find the size of the first chunk's embedding
    partial_embeddings = embedding_model.encode(list(chunks[0]))
    vector_size = len(partial_embeddings[0])

    # Set the default distance function, and catch for incorrect capitalization
    distance_function = distance_function.lower().capitalize()

    # Create a Qdrant client running locally in memory
    client = QdrantClient(':memory:')

    # Create a collection in the Qdrant client, and configure its vectors
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size, 
            distance=distance_function
            )
        )

    # Return the Qdrant client object
    return client


def upsert_embedding_db(
        client, 
        chunks, 
        embedding_model, 
        collection_name
        ) -> QdrantClient:
    '''
    Update/insert a Qdrant client's collection with the some chunks of text

    Args:
        client (QdrantClient): Initialized Qdrant client object

        chunks (List[str]): List of similar length strings to vectorize

        embedding (SentenceTransformer): SentenceTransformer object of your choice
        SentenceTransformer](https://huggingface.co/sentence-transformers)

        collection_name (str): Name of the collection being modified

    Returns:
        QdrantClient: Upserted Qdrant client object
    '''
    # Get the current count of chunks in the collection
    # TODO: the grossness here is a workaround for client.count() returning a unique
    # class object "count=0". If a method becomes available to get the count as an int,
    # this will be changed to use that method.
    current_count = int(str(client.count(collection_name)).partition('=')[-1])

    for id, chunk in enumerate(chunks):  # For each chunk
        # Embed the chunk
        embedded_chunk = list(embedding_model.encode(chunk))

        # Prepare the chunk as a list of (float) vectors
        prepped_chunk = [float(vector) for vector in embedded_chunk]

        # Create a payload of the (now embedded) chunk
        prepped_payload = {'chunk_string': chunk}

        # Upsert the embedded chunk and its payload into the collection
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=id + current_count,  # Make sure all chunks have sequential IDs
                    vector=prepped_chunk,
                    payload=prepped_payload
                    )
                ]
            )

    # Return the modified Qdrant client object
    return client
