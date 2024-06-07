# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.qdrant

'''
Vector databases embeddings using Qdrant: https://qdrant.tech/

See class `collection` docstring for a simple example, using the in-memory drive.

Example storing a Qdrant collection to disk:

```py
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from ogbujipt.text_helper import text_split
from ogbujipt.embedding.qdrant import collection

DBPATH = '/tmp/qdrant_test'
qclient = QdrantClient(path=DBPATH)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
collection = collection('my-text', embedding_model, db=qclient)

text = 'The quick brown fox\njumps over the lazy dog,\nthen hides under a log\nwith a frog.\n'
text += 'Should the hound wake up,\nall jumpers beware\nin a log, in a bog\nhe\'ll search everywhere.\n'
chunks = text_split(text, chunk_size=20, chunk_overlap=4, separator='\n')

collection.update(texts=chunks, metas=[{'seq-index': i} for (i, _) in enumerate(chunks)])
retval = collection.search('what does the fox say?', limit=1, score_threshold=0.5)
```

You can now always re-load the collections from that file via similar code in a different process

Refer to Qdrant docs: https://qdrant.github.io/qdrant/redoc/index.html
'''

import warnings
import itertools
# from typing import Sequence

# Qdrant install is optional for OgbujiPT
try:
    # pip install qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = object()  # Set up a dummy to satisfy the type hints
    models = None

# Option for running a Qdrant DB locally in memory
MEMORY_QDRANT_CONNECTION_PARAMS = {'location': ':memory:'}


class collection:
    def __init__(self, name, embedding_model, db=None,
                 distance_function=None, **conn_params):
        '''
        Initialize a Qdrant client

        Args:
            name (str): of the collection

            embedding (SentenceTransformer): SentenceTransformer object of your choice
            https://huggingface.co/sentence-transformers

            
            db (optional QdrantClient): existing DB/client to use, which should already be initialized

            distance_function (str): Distance function by which vectors will be compared

            conn_params (mapping): keyword parameters for setting up QdrantClient
            See the main docstring (or run `help(QdrantClient)`)
            https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/qdrant_client.py#L12

        Example:

        >>> from ogbujipt.text_helper import text_split
        >>> from ogbujipt.embedding.qdrant import collection  # pip install qdrant_client
        >>> from sentence_transformers import SentenceTransformer  # pip install sentence_transformers
        >>> text = 'The quick brown fox\njumps over the lazy dog,\nthen hides under a log\nwith a frog.\n'
        >>> text += 'Should the hound wake up,\nall jumpers beware\nin a log, in a bog\nhe\'ll search everywhere.\n'
        >>> embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        >>> collection = collection('my-text', embedding_model)
        >>> chunks = text_split(text, chunk_size=20, chunk_overlap=4, separator='\n')
        >>> collection.update(texts=chunks, metas=[{'seq-index': i} for (i, _) in enumerate(chunks)])
        >>> retval = collection.search('what does the fox say?', limit=1)
        '''
        self.name = name
        self.db = db
        # Check if the provided embedding model is a SentenceTransformer
        if embedding_model.__class__.__name__ == 'SentenceTransformer':
            self._embedding_model = embedding_model
        else:
            raise ValueError('embedding_model must be a SentenceTransformer object')

        if self.db:
            # Has the passed-in DB has been initialized?
            self._db_initialized = True
            try:
                self.db.get_collection(self.name)
            except ValueError:
                self._db_initialized = False
        elif not QDRANT_AVAILABLE: 
            raise RuntimeError('Qdrant not installed, you can run `pip install qdrant-client`')
        else:
            # Create a Qdrant client
            if not conn_params:
                conn_params = MEMORY_QDRANT_CONNECTION_PARAMS
            self.db = QdrantClient(**conn_params)
            self._db_initialized = False
        self._distance_function = distance_function or models.Distance.COSINE

    def _first_update_prep(self, text):
        if text.__class__.__name__ != 'str':
            raise ValueError('text must be a string')

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
            They'll be converted to embeddings for efficient lookup

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

            try:
                payload = dict(_text=text, **meta)
            except TypeError:
                raise TypeError('metas must be a list of mappings (dicts)')

            # Upsert the embedded chunk and its payload into the collection
            self.db.upsert(
                collection_name=self.name,
                points=[
                    models.PointStruct(
                        id=ix + before_count,  # sequential IDs
                        vector=embeddings,
                        payload=payload
                        )
                    ]
                )
    
    def reset(self):
        '''
        Reset the Qdrant collection, deleting the collection and all its contents
        '''
        if not self._db_initialized:
            raise RuntimeError('Qdrant Collection must be initialized before deleting its contents.')
        
        self.db.delete_collection(collection_name=self.name)
        self._db_initialized = False

    def search(self, query, **kwargs):
        '''
        Perform a search on this Qdrant collection

        Args:
            query (str): string to compare against items in the collection

            kwargs: other args to be passed to qdrant_client.QdrantClient.search(). Common ones:
                    limit - maximum number of results to return (useful for top-k query)
        '''
        if not self._db_initialized:
            warnings.warn('Qdrant Collection must be initialized. No contents.')
            return []
        
        if query.__class__.__name__ != 'str':
            raise ValueError('query must be a string')
        embedded_query = self._embedding_model.encode(query)
        return self.db.search(collection_name=self.name, query_vector=embedded_query, **kwargs)

    def count(self):
        '''
        Return the count of items in this Qdrant collection
        '''
        if not self._db_initialized:
            warnings.warn('Qdrant Collection must be initialized. No contents.')
            return 0
        # This ugly declaration just gets the count as an integer
        current_count = int(str(self.db.count(self.name)).partition('=')[-1])
        return current_count
