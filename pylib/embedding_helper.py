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

import warnings
import itertools

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = object()  # Set up a dummy to satisfy the type hints
    models = None

# Option for running a Qdrant DB locally in memory
MEMORY_QDRANT_CONNECTION_PARAMS = {'location': ':memory:'}


class qdrant_collection:
    def __init__(self, name, embedding_model, db=None,
                 distance_function=None, **conn_params):
        '''
        Initialize a Qdrant client

        Args:
            name (str): of the collection

            embedding (SentenceTransformer): SentenceTransformer object of your choice
            https://huggingface.co/sentence-transformers

            db (optional QdrantClient): existing DB/client to use

            distance_function (str): Distance function by which vectors will be compared

            conn_params (mapping): keyword parameters for setting up QdrantClient
            See the main docstring (or run `help(QdrantClient)`)
            https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/qdrant_client.py#L12

        Example:

        >>> from ogbujipt.text_helper import text_splitter
        >>> from ogbujipt.embedding_helper import qdrant_collection  # pip install qdrant_client
        >>> from sentence_transformers import SentenceTransformer  # pip install sentence_transformers
        >>> text = 'The quick brown fox\njumps over the lazy dog,\nthen hides under a log\nwith a frog.\n'
        >>> text += 'Should the hound wake up,\nall jumpers beware\nin a log, in a bog\nhe\'ll search everywhere.\n'
        >>> embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        >>> collection = qdrant_collection('my-text', embedding_model)
        >>> chunks = text_splitter(text, chunk_size=20, chunk_overlap=4, separator='\n')
        >>> collection.update(texts=chunks, metas=[{'seq-index': i} for (i, _) in enumerate(chunks)])
        >>> retval = collection.search('what does the fox say?', limit=1)
        retval
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
        self._distance_function = distance_function or models.Distance.COSINE
        self._db_initialized = False

    def _first_update_prep(self, text):
        # Make sure we have a vector size set; use a sample embedding if need be
        partial_embeddings = self._embedding_model.encode(text)
        self._vector_size = len(partial_embeddings)

        # Create a collection in the Qdrant client, and configure its vectors
        # Using REcreate_collection ensures overwrite for a clean, fresh, new collection
        self.db.recreate_collection(
            collection_name=self.name,
            vectors_config=models.VectorParams(
                size=self._vector_size, 
                distance=self._distance_function
                )
            )

        self._db_initialized = True

    def update(self, texts, metas=None):
        '''
        Update/insert into a Qdrant client's collection with the some chunks of text

        Args:
            texts (List[str]): Strings to be stored and indexed. For best results these should be of similar length.
                                They'll be converted to embeddings fo refficient lookup

            metas (List[dict]): Optional metadata per text, stored with the text and included whenever the text is
                                retrieved via search/query
        '''
        if len(texts) == 0:
            warnings.warn('Empty sequence of texts provided. No action will be taken.')
            return

        if metas is None:
            metas = [{}]*(len(texts))
        else:
            if len(texts) > len(metas):
                warnings.warn(f'More texts ({len(texts)} provided than metadata {len(metas)}).'
                              'Extra metadata items will be ignored.')
                metas = itertools.chain(metas, [{}]*(len(texts)-len(metas)))
            elif len(metas) > len(texts):
                warnings.warn(f'Fewer texts ({len(texts)} provided than metadata {len(metas)}). '
                              'The extra text will be given empty metadata.')
                metas = itertools.islice(metas, len(texts))

        if not self._db_initialized:
            self._first_update_prep(texts[0])
            before_count = 0
        else:
            before_count = self.count()

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
                        id=ix + before_count,  # Insistenmtly sequential IDs
                        vector=embeddings,
                        payload=payload
                        )
                    ]
                )

    def search(self, query, **kwargs):
        '''
        Perform a search on this Qdrant collection

        Args:
            query (str): string to compare against items in the collection

            kwargs: other args to be passed to qdrant_client.QdrantClient.search(). Common ones:
                    limit - maximum number of results to return (useful for top-k query)
        '''
        embedded_query = self._embedding_model.encode(query)
        return self.db.search(collection_name=self.name, query_vector=embedded_query, **kwargs)

    def count(self):
        '''
        Return the count of items in this Qdrant collection
        '''
        # This ugly declaration just gets the count as an integer
        current_count = int(str(self.db.count(self.name)).partition('=')[-1])
        return current_count
