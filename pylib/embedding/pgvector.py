# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.pgvector

'''
Vector databases embeddings using PGVector
'''

import warnings
import itertools
import json
from typing import Sequence

# Handle key imports
try:
    import asyncpg
    from pgvector.asyncpg import register_vector
    PREREQS_AVAILABLE = True
except ImportError:
    PREREQS_AVAILABLE = False
    asyncpg = None
    register_vector = object()  # Set up a dummy to satisfy the type hints
 
# ======================================================================================================================
# PG only supports proper query arguments (e.g. $1, $2, etc.) for values, not for table or column names
# Generic SQL template for creating a table to hold embedded documents
CREATE_DOC_TABLE = '''-- Create a table to hold embedded documents
CREATE TABLE IF NOT EXISTS {table_name} (
    id BIGSERIAL PRIMARY KEY,
    embedding VECTOR({embed_dimension}),  -- embedding vectors (array dimension)
    content TEXT NOT NULL,                -- text content of the chunk
    permission TEXT,                      -- permission of the chunk
    title TEXT,                           -- title of file
    page_numbers INTEGER[],               -- page number of the document that the chunk is found in
    tags TEXT[]                           -- tags associated with the chunk
);
'''

INSERT_DOCS = '''-- Insert a document into a table
INSERT INTO {table_name} (
    embedding,
    content,
    permission,
    title,
    page_numbers,
    tags
) VALUES ($1, $2, $3, $4, $5, $6);
'''

SEARCH_DOC_TABLE = '''-- Semantic search a document
SELECT 
    (embedding <=> '{search_embedding}') AS cosine_similarity,
    title,
    content,
    permission,
    page_numbers,
    tags
FROM 
    {table_name}
ORDER BY
    cosine_similarity ASC
LIMIT {limit};
'''
# ----------------------------------------------------------------------------------------------------------------------
# Generic SQL template for creating a table to hold individual messages from a chatlog and their metadata
CREATE_CHATLOG_TABLE = '''-- Create a table to hold individual messages from a chatlog and their metadata
CREATE TABLE IF NOT EXISTS {table_name} (
    id BIGSERIAL PRIMARY KEY,             -- unique id for the row
    history_key UUID,                     -- history key (unique identifier) of the chatlog this message belongs to
    index SERIAL,                         -- index of the message in the chatlog
    role INT,                             -- role of the message
    content TEXT NOT NULL,                -- text content of the message
    embedding VECTOR({embed_dimension}),  -- embedding vectors (array dimension)
    metadata_JSON JSON                    -- additional metadata of the message
);
'''

INSERT_CHATLOG = '''-- Insert a message into a chatlog
INSERT INTO {table_name} (
    history_key,
    index,
    role,
    content,
    embedding,
    metadata_JSON
) VALUES ($1, $2, $3, $4, $5, $6);
'''

RETURN_CHATLOG_BY_HISTORY_KEY = '''-- Get entire chatlog of a history key
SELECT
    index,
    role,
    content,
    metadata_JSON
FROM
    {table_name}
WHERE
    history_key = '{history_key}'
ORDER BY
    index ASC;
'''

SEMANTIC_SEARCH_CHATLOG_TABLE = '''-- Semantic search a chatlog
SELECT 
    (embedding <=> '{search_embedding}') AS cosine_similarity,
    index,
    role,
    content,
    metadata_JSON
FROM 
    {table_name}
WHERE
    history_key = '{history_key}'
ORDER BY
    cosine_similarity ASC
LIMIT {limit};
'''

# ======================================================================================================================

class PGvectorHelper:
    def __init__(self, embedding_model, table_name: str, apg_conn):
        '''
        Create a PGvector helper from an asyncpg connection

        If you don't yet have a connection, but have all the parameters,
        you can use the PGvectorHelper.from_conn_params() method instead

        Args:
            embedding (SentenceTransformer): SentenceTransformer object of your choice
            https://huggingface.co/sentence-transformers

            table_name: PostgresQL table to store the vector embeddings. Will be checked to restrict to
            alphanumeric characters and underscore

            apg_conn: asyncpg connection to the database
        '''
        if not PREREQS_AVAILABLE:
            raise RuntimeError('pgvector not installed, you can run `pip install pgvector asyncpg`')

        print(1, table_name)
        if not table_name.replace('_', '').isalnum():
            raise ValueError('table_name must be alphanumeric, with underscore also allowed')

        # Check if the provided embedding model is a SentenceTransformer
        if (embedding_model.__class__.__name__ == 'SentenceTransformer') and (not None):
            self._embedding_model = embedding_model
            self._embed_dimension = len(self._embedding_model.encode(''))
        elif embedding_model is None:
            self._embedding_model = None
            self._embed_dimension = 0
        else:
            raise ValueError('embedding_model must be a SentenceTransformer object or None')

        self.conn = apg_conn
        self.table_name = table_name

    @classmethod
    async def from_conn_params(cls, embedding_model, table_name,
                                user, password, db_name, host, port, **conn_params
            ) -> 'PGvectorHelper':
        '''
        Create a PGvector helper from connection parameters

        For details on accepted parameters, See the `pgvector_connection` docstring
        (e.g. run `help(pgvector_connection)`)
        '''
        try:
            conn = await asyncpg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=db_name,
                **conn_params
            )
        except Exception as e:
            # Don't blanket mask the exception. Handle exceptions types in whatever way makes sense
            raise
        return await cls.from_connection(embedding_model, table_name, conn)

    @classmethod
    async def from_connection(cls, embedding_model, table_name, conn) -> 'PGvectorHelper':
        '''
        Create a PGvector helper from connection parameters

        For details on accepted parameters, See the `pgvector_connection` docstring
        (e.g. run `help(pgvector_connection)`)
        '''
        # Ensure the vector extension is installed
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
        await register_vector(conn)
        # print('PGvector extension created and loaded.')
        return cls(embedding_model, table_name, conn)

    async def create_doc_table(self) -> None:
        '''
        Create the table to hold embedded documents
        '''
        # Check that the connection is still alive
        # if self.conn.is_closed():
        #     raise ConnectionError('Connection to database is closed')

        # Create the table
        await self.conn.execute(
            CREATE_DOC_TABLE.format(
                table_name=self.table_name,
                embed_dimension=self._embed_dimension)
            )
    
    async def insert_doc(
            self,
            content: str,
            permission: str = 'NULL',
            title: str = 'NULL',
            page_numbers: list[int] = [],
            tags: list[str] = []
        ) -> None:
        '''
        Update a table with one embedded document

        Args:
            content (str): text content of the document

            permission (str): permission of the document

            title (str): title of the document

            page_numbers (list): page number of the document that the chunk is found in

            tags (list): tags associated with the document
        '''
        # Get the embedding of the content as a PGvector compatible list
        content_embedding = self._embedding_model.encode(content)
        await self.conn.execute(INSERT_DOCS.format(table_name=self.table_name), content_embedding.tolist(), content,
                                permission, title, page_numbers, tags)

    async def insert_docs(
            self,
            content_list: Sequence[tuple[str, str | None,  str | None, list[int], list[str]]]
    ) -> None:
        '''
        Update a table with one or more embedded documents

        Semantically equivalent to multiple insert_doc calls, but uses executemany for efficiency

        Args:
            content_list (Sequence[ .. ]): A list of tuples, each of the form: (content, permission, title, page_numbers, tags)
        '''
        await self.conn.execute(INSERT_DOCS.format(table_name=self.table_name), [
            (self._embedding_model.encode(content), content, permission, title, page_numbers, tags)
            for content, permission, title, page_numbers, tags in content_list
        ])

    async def search_doc_table(
            self,
            query_string: str,
            limit: int = 1
            ) -> list[asyncpg.Record]:
        '''
        Similarity search documents using a query string

        Args:
            query_string (str): string to compare against items in the table

            k (int): maximum number of results to return (useful for top-k query)
        Returns:
            list[asyncpg.Record]: list of search results
            asyncpg.Record objects are similar to dicts, but allow for attribute-style access
        '''
        # Get the embedding of the query string as a PGvector compatible list
        query_embedding = list(self._embedding_model.encode(query_string))

        # Search the table
        # FIXME: Figure out the SQL injection guard for vectors. Not sure SQL Query params is an option here
        search_results = await self.conn.fetch(
            SEARCH_DOC_TABLE.format(
                table_name=self.table_name,
                search_embedding=query_embedding,
                limit=limit)
            )
        return search_results

    async def count_docs(self) -> int:
        '''
        Count the number of documents in the table

        Args:
        Returns:
            int: number of documents in the table
        '''
        # Count the number of documents in the table
        count = await self.conn.fetchval(f'SELECT COUNT(*) FROM {self.table_name}')
        return count

    async def drop_doc_table(self) -> None:
        '''
        Delete the table

        Exercise caution!
        '''
        # Delete the table
        await self.conn.execute(f'DROP TABLE IF EXISTS {self.table_name};')
        # await self.conn.execute(f'DROP TABLE {self.table_name}')