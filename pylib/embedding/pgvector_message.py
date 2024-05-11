# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.pgvector_message

'''
Vector embeddings DB feature for messaging (chat, etc.) using PGVector
'''

from uuid import UUID
from datetime import datetime, timezone
from typing import Iterable

from ogbujipt.config import attr_dict
from ogbujipt.embedding.pgvector import PGVectorHelper, asyncpg, process_search_response

__all__ = ['MessageDB']

# ------ SQL queries ---------------------------------------------------------------------------------------------------
# PG only supports proper query arguments (e.g. $1, $2, etc.) for values, not for table or column names
# Table names are checked to be legit sequel table names, and embed_dimension is assured to be an integer

CREATE_MESSAGE_TABLE = '''-- Create a table to hold individual messages (e.g. from a chatlog) and their metadata
CREATE TABLE IF NOT EXISTS {table_name} (
    ts TIMESTAMP WITH TIME ZONE,              -- timestamp of the message
    history_key UUID,                         -- uunique identifier for contextual message history
    role TEXT,                                -- role of the message (meta ID such as 'system' or user,
                                              -- or an ID associated with the sender)
    content TEXT NOT NULL,                    -- text content of the message
    embedding VECTOR({embed_dimension}),      -- embedding vectors (array dimension)
    metadata JSON                             -- additional metadata of the message
);
'''

INSERT_MESSAGE = '''-- Insert a message into a chatlog
INSERT INTO {table_name} (
    history_key,
    role,
    content,
    embedding,
    ts,
    metadata
) VALUES ($1, $2, $3, $4, $5, $6);
'''

CLEAR_MESSAGE = '''-- Deletes matching messages
DELETE FROM {table_name}
WHERE
    history_key = $1
'''

RETURN_MESSAGES_BY_HISTORY_KEY = '''-- Get entire chatlog by history key
SELECT
    ts,
    role,
    content,
    metadata
FROM
    {table_name}
WHERE
    history_key = $1
{since_clause}
ORDER BY
    ts DESC
{limit_clause};
'''

SEMANTIC_QUERY_MESSAGE_TABLE = '''-- Find messages with closest semantic similarity
SELECT
    1 - (embedding <=> $1) AS cosine_similarity,
    ts,
    role,
    content,
    metadata
FROM {table_name}
{where_clauses}
ORDER BY
    cosine_similarity DESC
{limit_clause};
'''

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

# Delete after full comfort with windowed implementation
# TEMPQ = '''
#         SELECT t_inner.ctid
#         FROM {table_name} t_inner
#         WHERE
#             t_inner.history_key = $1
#         ORDER BY
#             t_inner.ts DESC
#         LIMIT $2;
# '''

# ------ Class implementations ---------------------------------------------------------------------------------------

class MessageDB(PGVectorHelper):
    def __init__(self, embedding_model, table_name: str, pool: asyncpg.pool.Pool, window=0):
        '''
        Helper class for messages/chatlog storage and retrieval

        Args:
            embedding (SentenceTransformer): SentenceTransformer object of your choice
            https://huggingface.co/sentence-transformers
            window (int, optional): number of messages to maintain in the DB. Default is 0 (all messages)
        '''
        super().__init__(embedding_model, table_name, pool)
        self.window = window

    @classmethod
    async def from_conn_params(cls, embedding_model, table_name, host, port, db_name, user, password, window=0) -> 'MessageDB': # noqa: E501
        obj = await super().from_conn_params(embedding_model, table_name, host, port, db_name, user, password)
        obj.window = window
        return obj

    ''' Specialize PGvectorHelper for messages, e.g. chatlogs '''
    async def create_table(self):
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    CREATE_MESSAGE_TABLE.format(
                        table_name=self.table_name,
                        embed_dimension=self._embed_dimension
                    )
                )

    async def insert(
            self,
            history_key: UUID,
            role: str,
            content: str,
            # Timestamp later in order than SQL might suggest because of the defaulting
            timestamp: datetime | None = None,
            metadata: dict | None = None
    ) -> None:
        '''
        Update a table with one message's embedding

        Args:
            history_key (str): history key (unique identifier) of the chatlog this message belongs to

            role (str): role of the message (e.g. system, user, assistant, tool)

            content (str): text content of the message

            timestamp (datetime, optional): timestamp of the message, defaults to current time

            metadata (dict, optional): additional metadata of the message

        Returns:
            generator which yields the rows os the query results ass attributable dicts
        '''
        if not timestamp:
            timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)

        # Get the embedding of the content as a PGvector compatible list
        content_embedding = self._embedding_model.encode(content)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    INSERT_MESSAGE.format(table_name=self.table_name),
                    history_key,
                    role,
                    content,
                    content_embedding.tolist(),
                    timestamp,
                    metadata
                )
            # print(f'{self.window=}, Pre-count: {await self.count_items()}')
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                if self.window:
                    await conn.execute(
                        DELETE_OLDEST_MESSAGES.format(table_name=self.table_name),
                        history_key,
                        self.window)
        # async with self.pool.acquire() as conn:
        #     print(f'{self.window=}, Post-count: {await self.count_items()}, {list(await conn.fetch(TEMPQ.format(table_name=self.table_name), history_key, self.window))}')  # noqa E501

    async def insert_many(
            self,
            content_list: Iterable[tuple[str, list[str]]]
    ) -> None:
        '''
        Update table with one or (presumably) more message embedding

        Semantically equivalent to multiple insert calls, but uses executemany for efficiency

        Args:
            content_list: List of tuples, each of the form: (history_key, role, text, timestamp, metadata)
        '''
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    INSERT_MESSAGE.format(table_name=self.table_name),
                    (
                        (hk, role, text, self._embedding_model.encode(text), ts, metadata)
                        for hk, role, text, ts, metadata in content_list
                    )
                )
            # print(f'{self.window=}, Pre-count: {await self.count_items()}')  # noqa E501
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                if self.window:
                    # Set uniquifies the history keys
                    for hk in {hk for hk, _, _, _, _ in content_list}:
                        await conn.execute(
                            DELETE_OLDEST_MESSAGES.format(table_name=self.table_name),
                            hk, self.window)
        # async with self.pool.acquire() as conn:
        #     print(f'{self.window=}, {hk=}, Post-count: {await self.count_items()}, {list(await conn.fetch(TEMPQ.format(table_name=self.table_name), hk, self.window))}')  # noqa E501

    async def clear(
            self,
            history_key: UUID
    ) -> None:
        '''
        Remove all entries in a message history entries

        Args:
            history_key (str): history key (unique identifier) to match
        '''
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    CLEAR_MESSAGE.format(
                        table_name=self.table_name,
                    ),
                    history_key
                )

    # XXX: Change to a generator
    async def get_messages(
            self,
            history_key: UUID | str,
            since: datetime | None = None,
            limit: int = 0
    ): # -> list[asyncpg.Record]:
        '''
        Retrieve entries in a message history

        Args:
            history_key (str): history key (unique identifier) to match; string or object
            since (datetime, optional): only return messages after this timestamp
            limit (int, optional): maximum number of messages to return. Default is all messages
        Returns:
            generates asyncpg.Record instances of resulting messages
        '''
        if not isinstance(history_key, UUID):
            history_key = UUID(history_key)
            # msg = f'history_key must be a UUID, not {type(history_key)} ({history_key}))'
            # raise TypeError(msg)
        if not isinstance(since, datetime) and since is not None:
            msg = 'since must be a datetime or None'
            raise TypeError(msg)
        if not isinstance(limit, int):
            raise TypeError('limit must be an integer')  # Guard against injection

        qparams = [history_key]
        # Build query
        if since:
            # Don't really need the ${len(qparams) + N} thing here (first optional), but used for consistency
            since_clause = f' AND ts > ${len(qparams) + 1}'
            qparams.append(since)
        else:
            since_clause = ''
        if limit:
            limit_clause = f'LIMIT ${len(qparams) + 1}'
            qparams.append(limit)
        else:
            limit_clause = ''

        # Execute
        async with self.pool.acquire() as conn:
            message_records = await conn.fetch(
                RETURN_MESSAGES_BY_HISTORY_KEY.format(
                    table_name=self.table_name,
                    since_clause=since_clause,
                    limit_clause=limit_clause
                ),
                *qparams
            )

        return (attr_dict({
                'ts': record['ts'],
                'role': record['role'],
                'content': record['content'],
                'metadata': record['metadata']
            }) for record in message_records)

    async def search(
            self,
            history_key: UUID,
            text: str,
            since: datetime | None = None,
            threshold: float | None = None,
            limit: int = 1
    ) -> list[asyncpg.Record]:
        '''
        Similarity search documents using a query string

        Args:
            history_key (str): history key for the conversation to query
            text (str): string to compare against items in the table
            since (datetime, optional): only return results after this timestamp
            limit (int, optional): maximum number of messages to return; for top-k type query. Default is 1
        Returns:
            list[asyncpg.Record]: list of search results
                (asyncpg.Record objects are similar to dicts, but allow for attribute-style access)
        '''
        # Type checks
        if threshold is not None:
            if not isinstance(threshold, float) or (threshold < 0) or (threshold > 1):
                raise TypeError('threshold must be a float between 0.0 and 1.0')
        if not isinstance(limit, int):
            raise TypeError('limit must be an integer')
        if not isinstance(history_key, UUID):
            history_key = UUID(history_key)
        if not isinstance(since, datetime) and since is not None:
            msg = 'since must be a datetime or None'
            raise TypeError(msg)

        # Get embedding of the query string as a PGvector compatible list
        query_embedding = list(self._embedding_model.encode(text))

        # Build query
        clauses = ['main.history_key = $2\n']
        qparams = [query_embedding, history_key]
        if since is not None:
            # Don't really need the ${len(qparams) + N} thing here (first optional), but used for consistency
            clauses.append(f'ts > ${len(qparams) + 1}')
            qparams.append(since)
        if threshold is not None:
            clauses.append(THRESHOLD_WHERE_CLAUSE.format(query_threshold=f'${len(qparams) + 1}'))
            qparams.append(threshold)
        clauses = 'AND\n'.join(clauses)  # TODO: move this into the fstring below after py3.12
        where_clauses = f'WHERE\n{clauses}'
        limit_clause = f'LIMIT ${len(qparams) + 1}'
        qparams.append(limit)

        # Execute
        async with self.pool.acquire() as conn:
            message_records = await conn.fetch(
                SEMANTIC_QUERY_MESSAGE_TABLE.format(
                    table_name=self.table_name,
                    where_clauses=where_clauses,
                    limit_clause=limit_clause
                ),
                *qparams
            )

        search_results = [
            {
                'ts': record['ts'],
                'role': record['role'],
                'content': record['content'],
                'metadata': record['metadata'],
                'cosine_similarity': record['cosine_similarity']
            }
            for record in message_records
        ]

        return process_search_response(search_results)
