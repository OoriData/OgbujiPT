# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding_helper

'''
Helper for vector databases embeddings in such as Qdrant

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


class qdrant_collection:
    def __init__(self, name, embedding_model, db=None, **conn_params):
        '''
        Initialize a Qdrant client

        Args:
            name (str): of the collection

            embedding (SentenceTransformer): SentenceTransformer object of your choice
            https://huggingface.co/sentence-transformers

            db (optional QdrantClient): existing DB/client to use

            conn_params (mapping): keyword parameters for setting up QdrantClient
            See the main docstring (or run `help(QdrantClient)`)
            https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/qdrant_client.py#L12

        '''
        self.name = name
        self.db = db
        self._embedding_model = embedding_model
        if not self.db:
            if not QDRANT_AVAILABLE: 
                raise RuntimeError('Qdrant not installed, you can run `pip install qdrant-client`')

            # Create a Qdrant client
            if not conn_params:
                conn_params = MEMORY_QDRANT_CONNECTION_PARAMS
            self.db = QdrantClient(**conn_params)

    def add(self, texts, distance_function='Cosine',
            metas=None):
        '''
        Add a collection to a Qdrant client, and add some strings (chunks) to that collection

        Args:
            chunks (List[str]): List of similar length strings to embed

            distance_function (str): Distance function by which vectors will be compared

            qdrant_conn_params (mapping): keyword parameters for setting up QdrantClient
            See the main docstring (or run `help(QdrantClient)`)
            https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/qdrant_client.py#L12
        '''
        metas = metas or []
        # meta is a list of dicts
        # Find the size of the first chunk's embedding
        partial_embeddings = self._embedding_model.encode(texts[0])
        vector_size = len(partial_embeddings)

        # Set the default distance function, giving grace to capitalization
        distance_function = distance_function.lower().capitalize()

        # Create a collection in the Qdrant client, and configure its vectors
        # Using REcreate_collection ensures overwrite
        self.db.recreate_collection(
            collection_name=self.name,
            vectors_config=models.VectorParams(
                size=vector_size, 
                distance=distance_function
                )
            )

        # Put the items in the collection
        self.upsert(texts=texts, metas=metas)

    def upsert(self, texts, metas=None):
        '''
        Update/insert a Qdrant client's collection with the some chunks of text

        Args:
            texts (List[str]): Strings to be stored and indexed. For best results these should be of similar length.
                                They'll be converted to embeddings fo refficient lookup

            metas (List[dict]): Optional metadata per text, stored with the text and included whenever the text is
                                retrieved via search/query
        '''
        current_count = int(str(self.db.count(self.name)).partition('=')[-1])
        metas = metas or []

        for ix, (text, meta) in enumerate(zip(texts, metas)):
            # Embeddings as float/vectors
            # The inline prints actually turn into a cool progress indicator in jupyter 😁
            embeddings = list(map(float, self._embedding_model.encode(text)))

            payload = dict(_text=text, **meta)

            # Upsert the embedded chunk and its payload into the collection
            self.db.upsert(
                collection_name=self.name,
                points=[
                    models.PointStruct(
                        id=ix + current_count,  # Sequential IDs
                        vector=embeddings,
                        payload=payload
                        )
                    ]
                )

    def search(self, text, **kwargs):
        '''
        Perform a search on this Qdrant collection

        Args:
            query (str): string to compare against items in the collection

            kwargs: other args to be passed to qdrant_client.QdrantClient.search(). Common ones:
                    limit - maximum number of results to return (useful for top-k query)
        '''
        embedded_text = self._embedding_model.encode(text)
        return self.db.search(collection_name=self.name, query_vector=embedded_text, **kwargs)
