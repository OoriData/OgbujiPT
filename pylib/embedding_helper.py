# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding_helper

'''
Routines to help with embedding for vector databases such as Qdrant

Vector DBs are useful when you have a lot of context to use with LLMs,
e.g. a large document or collection of docs. One common pattern is to create
vector indices on this text. Given an LLM prompt, the vector DB is first queried
to find the most relevant "top k" sections of text based on the prompt,
which are added as context in the ultimate LLM invocation.

For sample code see demo/chat_pdf_streamlit_ui.py

You need an LLM to turn the text into vectors for such indexing, and these
vectors are called the embeddings. You can usually create useful embeddings
with a less powerful (and more efficient) LLM.

This module provides utilities to set up a vector DB, and use it to index
chunks of text using a provided LLM model to create the embeddings.

Other common use-cases for vector DBs in LLM applications:

* Long-term LLM Memory for chat: index the entire chat history and retrieve
the most relevant and recent N messages based on the user's new message,
to give the LLM a chance to make its responses coherent and on-topic

* Cache previous LLM interactions, saving resources by retrieving previous
responses to similar questions without having to use the most powerful LLM
'''

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = object()  # Set up a dummy to satisfy the type hints


# Option for running a Qdrant DB locally in memory
MEMORY_QDRANT_CONNECTION_PARAMS = {'location': ':memory:'}


def qdrant_init_embedding_db(**qdrant_conn_params) -> QdrantClient:
    '''
    Initialize a Qdrant client

    Args:
        qdrant_conn_params (mapping): keyword parameters for setting up QdrantClient
        See the main docstring (or run `help(QdrantClient)`)
        https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/qdrant_client.py#L12

    Returns:
        QdrantClient: Qdrant client object
    '''
    if not QDRANT_AVAILABLE: 
        raise RuntimeError('Qdrant not installed, you can run `pip install qdrant-client`')
    
    # Create a Qdrant client
    if not qdrant_conn_params:
        qdrant_conn_params = MEMORY_QDRANT_CONNECTION_PARAMS
    client = QdrantClient(**qdrant_conn_params)

    # Return the Qdrant client object
    return client


def qdrant_add_collection(
        client,
        chunks, 
        embedding_model, 
        collection_name, 
        distance_function='Cosine'
        ) -> QdrantClient:
    '''
    Add a collection to a Qdrant client, and add some strings (chunks) to that collection

    Args:
        client (QdrantClient): Initialized Qdrant client object

        chunks (List[str]): List of similar length strings to embed

        embedding (SentenceTransformer): SentenceTransformer object of your choice
        https://huggingface.co/sentence-transformers

        collection_name (str): Name that describes "chunks"

        distance_function (str): Distance function by which vectors will be compared

        qdrant_conn_params (mapping): keyword parameters for setting up QdrantClient
        See the main docstring (or run `help(QdrantClient)`)
        https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/qdrant_client.py#L12

    Returns:
        QdrantClient: Qdrant client object with new collection
    '''
    if not QDRANT_AVAILABLE:
        raise RuntimeError('Qdrant not installed, you can run `pip install qdrant-client`')
    
    # Find the size of the first chunk's embedding
    partial_embeddings = embedding_model.encode(chunks[0])
    vector_size = len(partial_embeddings)

    # Set the default distance function, and catch for incorrect capitalization
    distance_function = distance_function.lower().capitalize()

    ## Create a collection in the Qdrant client, and configure its vectors
    # Using REcreate_collection ensures overwrite
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size, 
            distance=distance_function
            )
        )
    
    # Put the chunks in the collection
    client = qdrant_upsert_collection(
        client=client,
        chunks=chunks,
        embedding_model=embedding_model,
        collection_name=collection_name
        )

    # Return the Qdrant client object
    return client


def qdrant_upsert_collection(
        client, 
        chunks, 
        embedding_model, 
        collection_name
        ) -> QdrantClient:
    '''
    Update/insert a Qdrant client's collection with the some chunks of text

    Args:
        client (QdrantClient): Initialized Qdrant client object

        chunks (List[str]): List of similar length strings to embed

        embedding (SentenceTransformer): SentenceTransformer object of your choice
        SentenceTransformer](https://huggingface.co/sentence-transformers)

        collection_name (str): Name of the collection being modified

    Returns:
        QdrantClient: Upserted Qdrant client object
    '''
    if not QDRANT_AVAILABLE:
        raise RuntimeError('Qdrant not installed, you can run `pip install qdrant-client`')

    # Get the current count of chunks in the collection
    # TODO: the grossness here is a workaround for client.count() returning
    # an object which can then be cast to a string such as "count=0"
    # We'll prefer to use a method to get the count directly as an int,
    # once one becomes available
    current_count = int(str(client.count(collection_name)).partition('=')[-1])

    for ix, chunk in enumerate(chunks):  # For each chunk
        # Embeddings as float/vectors
        embeddings = list(map(float, embedding_model.encode(chunk)))

        # Create a payload of the (now embedded) chunk
        prepped_payload = {'chunk_string': chunk}

        # Upsert the embedded chunk and its payload into the collection
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=ix + current_count,  # Make sure all chunks have sequential IDs
                    vector=embeddings,
                    payload=prepped_payload
                    )
                ]
            )

    # Return the modified Qdrant client object
    return client
