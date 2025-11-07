# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.memoization.pgmemo
'''
Memoization using PGVector

Resources
* https://www.postgresql.org/docs/current/pgtrgm.html
'''

import json
# import asyncio
# from typing import ClassVar
from functools import partial

# from ogbujipt.embedding.pgvector import process_search_response
from ogbujipt.embedding.pgvector import (asyncpg, DEFAULT_SCHEMA, CHECK_TABLE_EXISTS,
                                         DEFAULT_MIN_CONNECTION_POOL_SIZE, DEFAULT_MAX_CONNECTION_POOL_SIZE)


# ------ SQL queries ---------------------------------------------------------------------------------------------------
# PG only supports proper query arguments (e.g. $1, $2, etc.) for values, not for table or column names
# Table names are checked to be legit sequel table names, and embed_dimension is assured to be an integer

CREATE_MEMO_TABLE = '''-- Create a table to hold memos (i.e. prior LLM requests which match to a high degree)
CREATE TABLE IF NOT EXISTS {table_name} (
    ts TIMESTAMP WITH TIME ZONE,              -- timestamp of the entry; can be used for aging
    llm_id TEXT NOT NULL,                     -- Identifies the LLM which produced this prompt/response pair
    prompt TEXT NOT NULL,                     -- LLM completion prompt; usually a tokenized chat prompt
    response TEXT NOT NULL,                   -- Response from LLM
    metadata JSON                             -- additional metadata for the entry
);
'''

INSERT_ENTRY = '''-- Record an entry
INSERT INTO {table_name} (
    llm_id,
    prompt,
    response,
    ts,
    metadata
) VALUES ($1, $2, $3, $4, $5);
'''

CLEAR_BY_LLM = '''-- Removes responses from a given LLM
DELETE FROM {table_name}
WHERE
    llm_id = $1
'''

# TODO: Clear by age

CHECK_MEMO = '''-- Check a new prompt to see if it can be considered a repeat
SELECT
    response,
    ts,
    metadata
FROM
    {table_name}
WHERE
    llm_id = $1
    AND
    prompt <-> $2 < $3  -- Prompt's distance is low enough from input
'''

# TODO: BELOW

# The cosine_similarity alias is not available in the WHERE clause, so use a nested SELECT
SEMANTIC_QUERY_MESSAGE_TABLE = '''-- Find messages with closest semantic similarity
SELECT
    cosine_similarity,
    ts,
    role,
    content,
    metadata
FROM
    (SELECT
        history_key,
        1 - (embedding <=> $1) AS cosine_similarity,
        ts,
        role,
        content,
        metadata FROM {table_name}) AS main
{where_clauses}
{limit_clause};
'''

THRESHOLD_WHERE_CLAUSE = 'main.cosine_similarity >= {query_threshold}\n'

DELETE_OLDEST_MESSAGES = '''-- Delete oldest messages for given history key, such that only the newest N messages remain
DELETE FROM {table_name} t_outer
WHERE
    t_outer.history_key = $1
AND
    t_outer.ctid NOT IN (
        SELECT t_inner.ctid
        FROM {table_name} t_inner
        WHERE
            t_inner.history_key = $1
        ORDER BY
            t_inner.ts DESC
        LIMIT $2
);
'''

class PGMemo:
    '''
    Helper class for memoizing LLM requests in Postgres

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
    def __init__(self, table_name: str, pool, schema=None):
        '''
        If you don't already have a connection pool, construct using the PGMemo.from_pool_params() method

        Args:
            table_name: PostgresQL table. Checked to restrict to alphanumeric characters, underscores & periods

            pool: asyncpg connection pool instance (asyncpg.pool.Pool)

            schema: a schema to which the JSON extension has been set. In more sophisticated DB setups
                using multiple schemata, you can run into `ERROR: type "JSON" does not exist`
                unless a schema with the extension is in the search path (via `SET SCHEMA`)
        '''
        if not table_name.replace('_', '').replace('.', '').isalnum():
            msg = f'table_name must be alphanumeric, with optional underscore or periods. Got: {table_name}'
            raise ValueError(msg)

        self.table_name = table_name
        self.pool = pool
        self.schema = schema

    @classmethod
    async def from_conn_params(cls, table_name, host, port, db_name, user, password,
        schema=None, pool_min=DEFAULT_MIN_CONNECTION_POOL_SIZE, pool_max=DEFAULT_MAX_CONNECTION_POOL_SIZE) -> 'PGMemo': # noqa: E501
        '''
        Create PGMemo instance from connection/pool parameters

        Will create a connection pool, with JSON type handling initialized,
        and set that as a pool attribute on the created object as a user convenience.

        For details on accepted parameters, See the class docstring
            (e.g. run `help(PGMemo)`)
        '''
        init_pool_ = partial(PGMemo.init_pool, schema=schema)
        pool = await asyncpg.create_pool(init=init_pool_, host=host, port=port, user=user,
                    password=password, database=db_name, min_size=pool_min, max_size=pool_max)

        new_obj = cls(table_name, pool, schema=schema)
        return new_obj

    @classmethod
    async def from_conn_string(cls, conn_string, table_name,
         schema=None, pool_min=DEFAULT_MIN_CONNECTION_POOL_SIZE, pool_max=DEFAULT_MAX_CONNECTION_POOL_SIZE) -> 'PGMemo': # noqa: E501
        '''
        Create PGMemo instance from a connection string AKA DSN

        Will create a connection pool, with JSON type handling initialized,
        and set that as a pool attribute on the created object as a user convenience.
        '''
        # https://github.com/MagicStack/asyncpg/blob/0a322a2e4ca1c3c3cf6c2cf22b236a6da6c61680/asyncpg/pool.py#L339
        init_pool_ = partial(PGMemo.init_pool, schema=schema)
        pool = await asyncpg.create_pool(
            conn_string, init=init_pool_, min_size=pool_min, max_size=pool_max)

        new_obj = cls(table_name, pool, schema=schema)
        return new_obj

    @staticmethod
    async def init_pool(conn, schema=None):
        '''
        Initialize vector extension for a connection from a pool

        Can be invoked from upstream if they're managing the connection pool themselves

        If they choose to have us create a connection pool (e.g. from_conn_params), it will use this
        '''
        schema = schema or DEFAULT_SCHEMA
        # TODO: Clean all this up
        try:
            await conn.set_type_codec(  # Register a codec for JSON
                'JSON', encoder=json.dumps, decoder=json.loads, schema=schema)
        except ValueError:
            try:
                await conn.set_type_codec(  # Register a codec for JSON
                    'JSON', encoder=json.dumps, decoder=json.loads)
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

    async def create_table(self) -> None:
        '''
        Create the memo table if it doesn't exist.
        
        Also ensures the pg_trgm extension is enabled for trigram similarity matching.
        '''
        async with self.pool.acquire() as conn:
            # Enable pg_trgm extension for trigram similarity
            await conn.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm;')
            # Create the table
            await conn.execute(CREATE_MEMO_TABLE.format(table_name=self.table_name))
            # Create index on prompt for faster similarity searches
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS {self.table_name}_prompt_trgm_idx '
                f'ON {self.table_name} USING gin (prompt gin_trgm_ops);'
            )

    async def check_memo(self, llm_id: str, prompt: str, similarity_threshold: float = 0.1) -> dict | None:
        '''
        Check if a prompt is similar enough to a previous one to use a cached response.
        
        Uses PostgreSQL trigram similarity (pg_trgm extension) to find similar prompts.
        The <-> operator returns a distance (0 = identical, higher = more different).
        
        Args:
            llm_id: Identifier for the LLM model
            prompt: The prompt text to check
            similarity_threshold: Maximum distance threshold (lower = more similar required).
                Default 0.1 means prompts must be very similar. Typical range: 0.05-0.3
        
        Returns:
            dict with keys 'response', 'ts', 'metadata' if a match is found, None otherwise
        '''
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                CHECK_MEMO.format(table_name=self.table_name),
                llm_id, prompt, similarity_threshold
            )
            if result:
                return {
                    'response': result['response'],
                    'ts': result['ts'],
                    'metadata': result['metadata'] if result['metadata'] else {}
                }
            return None

    async def insert_entry(self, llm_id: str, prompt: str, response: str, metadata: dict | None = None) -> None:
        '''
        Insert a new memo entry (prompt/response pair).
        
        Args:
            llm_id: Identifier for the LLM model
            prompt: The prompt text
            response: The LLM response text
            metadata: Optional metadata dictionary (will be stored as JSON)
        '''
        from datetime import datetime
        async with self.pool.acquire() as conn:
            # Convert metadata to JSON string if provided, otherwise use None
            metadata_json = json.dumps(metadata) if metadata else None
            await conn.execute(
                INSERT_ENTRY.format(table_name=self.table_name),
                llm_id, prompt, response, datetime.now(), metadata_json
            )

    async def clear_by_llm(self, llm_id: str) -> int:
        '''
        Remove all memo entries for a given LLM.
        
        Args:
            llm_id: Identifier for the LLM model
            
        Returns:
            Number of rows deleted
        '''
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                CLEAR_BY_LLM.format(table_name=self.table_name),
                llm_id
            )
            # Extract number of rows deleted from result string like "DELETE 5"
            return int(result.split()[-1]) if result else 0
