# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.pgvector

'''
Vector databases embeddings using PGVector
'''

import json

from ogbujipt.config import attr_dict

# Handle key imports
try:
    import asyncpg
    from pgvector.asyncpg import register_vector
    PREREQS_AVAILABLE = True
except ImportError:
    PREREQS_AVAILABLE = False
    asyncpg = None
    register_vector = object()  # Set up a dummy to satisfy the type hints

# ------ SQL queries ---------------------------------------------------------------------------------------------------
# PG only supports proper query arguments (e.g. $1, $2, etc.) for values, not for table or column names
# Table names are checked to be legit sequel table names, and embed_dimension is assured to be an integer

CREATE_VECTOR_EXTENSION = 'CREATE EXTENSION IF NOT EXISTS vector;'

CHECK_TABLE_EXISTS = '''-- Check if a table exists
SELECT EXISTS (
    SELECT FROM pg_tables
    WHERE tablename = $1
);
'''
# ------ SQL queries ---------------------------------------------------------------------------------------------------

# Default overall PG max is 100.
# See: https://commandprompt.com/education/how-to-alter-max_connections-parameter-in-postgresql/
DEFAULT_MIN_MAX_CONNECTION_POOL_SIZE = (10, 20)


class PGVectorHelper:
    def __init__(self, embedding_model, table_name: str, apg_conn_pool):
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

        self.conn_pool = apg_conn_pool
        self.table_name = table_name

    @classmethod
    async def from_conn_params(
            cls,
            embedding_model,
            table_name,
            user, 
            password,
            db_name,
            host,
            port,
            min_max_size=DEFAULT_MIN_MAX_CONNECTION_POOL_SIZE,
            **conn_params
    ) -> 'PGVectorHelper':
        '''
        Create a PGvector helper from connection parameters

        For details on accepted parameters, See the `pgvector_connection` docstring
            (e.g. run `help(pgvector_connection)`)
        '''
        min_size, max_size = min_max_size
        # FIXME: Clean up this exception handling
        # try:
        # import logging
        # logging.critical(f'Connecting to {host}:{port} as {user} to {db_name}')
        # logging.critical(str(conn_params))
        conn_pool = await asyncpg.create_pool(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name,
            min_size=min_size,
            max_size=max_size,
            **conn_params
        )
        # except Exception as e:
            # Don't blanket mask the exception. Handle exceptions types in whatever way makes sense
            # raise e
        return await cls.from_connection_pool(embedding_model, table_name, conn_pool)

    @classmethod
    async def from_connection_pool(cls, embedding_model, table_name, conn_pool) -> 'PGVectorHelper':
        '''
        Create a PGvector helper from a connection pool instance

        For details on accepted parameters, See the `pgvector_connection` docstring
            (e.g. run `help(pgvector_connection)`)
        '''
        # Ensure the vector extension is installed
        async with conn_pool.acquire() as conn:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
            await register_vector(conn)

            await conn.set_type_codec(  # Register a codec for JSON
                'JSON',
                encoder=json.dumps,
                decoder=json.loads,
                schema='pg_catalog'
            )

        # print('PGvector extension created and loaded.')
        return cls(embedding_model, table_name, conn_pool)

    # Hmm. Just called count in the qdrant version
    async def count_items(self) -> int:
        '''
        Count the number of documents in the table

        Returns:
            int: number of documents in the table
        '''
        # Count the number of documents in the table
        async with self.conn_pool.acquire() as conn:
            count = await conn.fetchval(f'SELECT COUNT(*) FROM {self.table_name}')
        return count
    
    async def table_exists(self) -> bool:
        '''
        Check if the table exists

        Returns:
            bool: True if the table exists, False otherwise
        '''
        # Check if the table exists
        async with self.conn_pool.acquire() as conn:
            table_exists = await conn.fetchval(
                CHECK_TABLE_EXISTS,
                self.table_name
            )
        return table_exists

    async def drop_table(self) -> None:
        '''
        Delete the table

        Exercise caution!
        '''
        # Delete the table
        async with self.conn_pool.acquire() as conn:
            await conn.execute(f'DROP TABLE IF EXISTS {self.table_name};')


def process_search_response(qresponse):
    '''
    Convert a query response to an attributable dict

    Args:
        query_response (asyncpg.Record): asyncpg.Record object to be converted to a dict

    Returns:
        list[dict]: List with a dict representation for each result row

    >>> await mydb.search(text='Hello')
    >>> results = process_search_response()
    >>> row = next(results)  # Assume there's at least one result
    >>> c = r.content
    >>> t = r.tags

    If a row does not have a title or page_numbers field, these will be set to None

    Other reasons for this conversion: asyncpg.Record objects are not JSON serializable,
    and don't support attribute-style access
    '''
    for row in qresponse:
        # Actually, this is otiose; just let missing attributes fail
        # if 'title' not in row:
        #     row['title'] = None
        # if 'page_numbers' not in row:
        #     row['page_numbers'] = None
        # print(row, row.items())
        yield attr_dict(row)


# Down here to avoid circular imports
from ogbujipt.embedding.pgvector_data_doc import DataDB, DocDB  # noqa: E402 F401
from ogbujipt.embedding.pgvector_chat import MessageDB  # noqa: E402 F401
