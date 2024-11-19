# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.pgvector
'''
Vector databases embeddings using PGVector (https://github.com/pgvector/pgvector)

Includes features up to pgvector 0.7.0 (e.g. quantized vectors & indices)

Jonathan Katz has a very lucid analysis of the vector quantization features in 0.7.0, published pre-release
https://jkatz05.com/post/postgres/pgvector-scalar-binary-quantization/
'''
import json
# import asyncio
# from typing import ClassVar
# from enum import Enum
from functools import partial

from ogbujipt.config import attr_dict

# Handle key imports
try:
    import asyncpg
    from pgvector.asyncpg import register_vector
    PREREQS_AVAILABLE = True
    POOL_TYPE = asyncpg.pool.Pool
except ImportError as e:
    import warnings
    warnings.warn(f'Missing module {e.name}; required for using PGVector')
    PREREQS_AVAILABLE = False
    asyncpg = None
    register_vector = object()  # Set up a dummy to satisfy the type hints
    POOL_TYPE = object

# See: https://github.com/OoriData/OgbujiPT/issues/87
DEFAULT_USER_SCHEMA = 'public'
DEFAULT_SYSTEM_SCHEMA = 'pg_catalog'

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
DEFAULT_MIN_CONNECTION_POOL_SIZE = 10
DEFAULT_MAX_CONNECTION_POOL_SIZE = 20


class PGVectorHelper:
    '''
    Helper class for PGVector operations

    Construct using from_conn_params() or from_conn_string() class method

    Connection and pool parameters:

    * table_name: PostgresQL table name. Restricted to alphanumeric characters, underscore & period (for schema qual)
    * host: Hostname or IP address of the PostgreSQL server. Defaults to UNIX socket if not provided.
    * port: Port number at which the PostgreSQL server is listening. Defaults to 5432 if not provided.
    * user: User name used to authenticate.
    * password: Password used to authenticate.
    * database: Database name to connect to.
    * pool_min: minimum number of connections to maintain in the pool (used as min_size for create_pool).
    * pool_max: maximum number of connections to maintain in the pool (used as max_size for create_pool).
    '''
    def __init__(self, embedding_model, table_name: str, pool, stringify_json=False, sys_schema=DEFAULT_SYSTEM_SCHEMA,
                 half_precision=False, itypes=None, ifuncs=None, i_max_conn=16, ef_construction=64):
        '''
        If you don't already have a connection pool, construct using the PGvectorHelper.from_pool_params() method

        Args:
            embedding (SentenceTransformer): SentenceTransformer object of your choice
            https://huggingface.co/sentence-transformers

            table_name (str): PostgresQL table. Checked to restrict to alphanumeric characters & underscore

            pool (asyncpg.pool.Pool): asyncpg connection pool instance

            stringify_json (bool): if True, interchange JSON metadata to/from string,
                rather than relying on the registered data type. Defaults to False.

            sys_schema (str): schema to which the vector extension has been set. In more sophisticated DB setups
                using multiple schemata, you can run into `ERROR: type "vector" does not exist`
                unless a schema with the extension is in the search path (via `SET SCHEMA`)

            half_precision (bool): if True, use halfvec type to store half-precision vectors (pgvector 0.7.0+ only).
                Default is False (full precision)

            itypes (list <str>): Index types (or empty for no indexing).
                An index will be build for each combo of itypes & ifunc
                Values: vector, halfvec (0.7.0 & up), bit, sparsevec (0.7.0+)

            ifuncs (list <str>): Index functions (defaults to ['cosine'] if itypes is not None)
                An index will be build for each combo of itypes & ifunc
                Values: l2, ip, cosine, l1, hamming (0.7.0+, with bit type only), jaccard (0.7.0+, with bit type only)

            i_max_conn (int): max number of connections per layer (16 by default);
                see pgvector docs

            ef_construction (int): size of the dynamic candidate list for constructing the graph (64 by default);
                see pgvector docs
        '''
        if not PREREQS_AVAILABLE:
            raise RuntimeError('pgvector not installed, you can run `pip install pgvector asyncpg`')

        if not table_name.replace('_', '').replace('.', '').isalnum():
            msg = f'table_name must be alphanumeric, with optional underscore or periods. Got: {table_name}'
            raise ValueError(msg)

        self.table_name = table_name
        self.embedding_model = embedding_model
        self.pool = pool
        self.stringify_json = stringify_json
        self.sys_schema = sys_schema
        if half_precision:
            self.vtype = 'halfvec'
        else:
            self.vtype = 'vector'
        itypes = itypes or []
        assert all([t in ('vector', 'halfvec', 'bit', 'sparsevec') for t in itypes])
        self.itypes = itypes
        ifuncs = ifuncs or []
        if itypes and not ifuncs:
            ifuncs = ['cosine']
        assert all([f in ('l2', 'ip', 'cosine', 'l1', 'hamming', 'jaccard') for f in ifuncs])
        self.ifuncs = ifuncs
        self.i_max_conn = i_max_conn
        self.ef_construction = ef_construction

        # Check if the provided embedding model is a SentenceTransformer
        if (embedding_model.__class__.__name__ == 'SentenceTransformer') and (not None):
            self._embedding_model = embedding_model
            self._embed_dimension = len(self._embedding_model.encode(''))
        elif embedding_model is None:
            self._embedding_model = None
            self._embed_dimension = 0
        else:
            raise ValueError('embedding_model must be a SentenceTransformer object or None')

    @classmethod
    async def from_conn_params(cls, embedding_model, table_name, host, port, db_name, user, password,
        stringify_json=False, sys_schema=DEFAULT_SYSTEM_SCHEMA, pool_min=DEFAULT_MIN_CONNECTION_POOL_SIZE,
        pool_max=DEFAULT_MAX_CONNECTION_POOL_SIZE, half_precision=False, itypes=None, ifuncs=None,
        i_max_conn=16, ef_construction=64) -> 'PGVectorHelper': # noqa: E501
        '''
        Create PGVectorHelper instance from connection/pool parameters

        Will create a connection pool, with JSON type handling initialized,
        and set that as a pool attribute on the created object as a user convenience.

        For details on accepted parameters, See the class docstring
            (e.g. run `help(PGVectorHelper)`)
        '''
        if not sys_schema.replace('_', '').replace('.', '').isalnum():
            msg = f'System schema must be alphanumeric, with optional underscore or periods. Got: {sys_schema}'
            raise ValueError(msg)

        # https://github.com/MagicStack/asyncpg/blob/0a322a2e4ca1c3c3cf6c2cf22b236a6da6c61680/asyncpg/pool.py#L339
        init_pool_ = partial(PGVectorHelper.init_pool, schema=sys_schema)
        pool = await asyncpg.create_pool(init=init_pool_, host=host, port=port, user=user,
                    password=password, database=db_name, min_size=pool_min, max_size=pool_max)

        new_obj = cls(embedding_model, table_name, pool, stringify_json=stringify_json, sys_schema=sys_schema,
                        half_precision=half_precision, itypes=itypes, ifuncs=ifuncs,
                        i_max_conn=i_max_conn, ef_construction=ef_construction)
        return new_obj

    @classmethod
    async def from_conn_string(cls, conn_string, embedding_model, table_name,
        stringify_json=False, sys_schema=DEFAULT_SYSTEM_SCHEMA, pool_min=DEFAULT_MIN_CONNECTION_POOL_SIZE,
        pool_max=DEFAULT_MAX_CONNECTION_POOL_SIZE, half_precision=False,
        itypes=None, ifuncs=None, i_max_conn=16, ef_construction=64) -> 'PGVectorHelper': # noqa: E501
        '''
        Create PGVectorHelper instance from a connection string AKA DSN

        Will create a connection pool, with JSON type handling initialized,
        and set that as a pool attribute on the created object as a user convenience.
        '''
        if not sys_schema.replace('_', '').replace('.', '').isalnum():
            msg = f'System schema must be alphanumeric, with optional underscore or periods. Got: {sys_schema}'
            raise ValueError(msg)

        init_pool_ = partial(PGVectorHelper.init_pool, schema=sys_schema)
        pool = await asyncpg.create_pool(
            conn_string, init=init_pool_, min_size=pool_min, max_size=pool_max)

        new_obj = cls(embedding_model, table_name, pool, stringify_json=stringify_json, sys_schema=sys_schema,
                        half_precision=half_precision, itypes=itypes, ifuncs=ifuncs,
                        i_max_conn=i_max_conn, ef_construction=ef_construction)
        return new_obj

    # Note the system schema default being None here. that's just to concord with how pgvector works
    # If a schema is provided, it will be used
    @staticmethod
    async def init_pool(conn, schema=DEFAULT_SYSTEM_SCHEMA):
        '''
        Initialize vector extension for a connection from a pool

        Can be invoked from upstream if they're managing the connection pool themselves

        If they choose to have us create a connection pool (e.g. from_conn_params), it will use this
        '''
        await conn.execute(f'CREATE EXTENSION IF NOT EXISTS vector SCHEMA {schema};')
        try:
            await register_vector(conn, schema=schema)
        except ValueError as e:
            raise RuntimeError(f'Unable to find vector type in the DB/schema. You might need to enable it: {e}')
        try:
            await conn.set_type_codec(  # Register a codec for JSON
                'jsonb', encoder=json.dumps, decoder=json.loads, schema=schema)
        except ValueError:
            try:
                await conn.set_type_codec(  # Register a codec for JSON
                    'jsonb', encoder=json.dumps, decoder=json.loads)
            except ValueError:
                pass

    # Hmm. Just called count in the qdrant version
    async def count_items(self) -> int:
        '''
        Count the number of documents in the table

        Returns:
            int: number of documents in the table
        '''
        async with self.pool.acquire() as conn:
            # Count the number of documents in the table
            count = await conn.fetchval(f'SELECT COUNT(*) FROM {self.table_name}')
        return count

    async def table_exists(self) -> bool:
        '''
        Check if the table exists

        Returns:
            bool: True if the table exists, False otherwise
        '''
        # Check if the table exists
        async with self.pool.acquire() as conn:
            exists = await conn.fetchval(
                CHECK_TABLE_EXISTS,
                self.table_name
            )
        return exists

    async def drop_table(self) -> None:
        '''
        Delete the table

        Exercise caution!
        '''
        # Delete the table
        async with self.pool.acquire() as conn:
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


# PG JSON operators https://www.postgresql.org/docs/16/functions-json.html
# More tutorial-like: https://popsql.com/learn-sql/postgresql/how-to-query-a-json-column-in-postgresql
def match_exact(key, val):
    '''
    Filter specifier to only return rows where the given top-level key exists in metadata, and matches the given value
    '''
    assert key.replace('_', '').replace('-', '').isalnum(), 'key contains nonalphanumeric, "-", or "_" characters'
    if isinstance(val, str):
        cast = ''
    elif isinstance(val, bool):
        cast = '::boolean'
    elif isinstance(val, int):
        cast = '::int'
    elif isinstance(val, float):
        cast = '::float'
    def apply():
        return f'(metadata ->> \'{key}\'){cast} = ${{}}', val
    return apply


def match_oneof(key, options: tuple[str]):
    '''
    Filter specifier to only return rows where the given top-level key exists in metadata,
    and matches one of the given values
    '''
    options = tuple(options)
    assert options
    assert key.replace('_', '').replace('-', '').isalnum(), 'key contains nonalphanumeric, "-", or "_" characters'
    option1 = options[0]
    if isinstance(option1, str):
        cast = ''
    if isinstance(option1, bool):
        cast = '::boolean'
    elif isinstance(option1, int):
        cast = '::int'
    elif isinstance(option1, float):
        cast = '::float'
    def apply():
        # return f'(metadata ->> \'{key}\'){cast} IN ${{}}', options
        return f'(metadata ->> \'{key}\'){cast} = ANY(${{}})', options
    return apply


# Down here to avoid circular imports
from ogbujipt.embedding.pgvector_data import DataDB  # noqa: E402 F401
from ogbujipt.embedding.pgvector_message import MessageDB  # noqa: E402 F401
