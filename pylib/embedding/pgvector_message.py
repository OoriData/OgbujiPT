# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.pgvector_message

'''
Vector embeddings DB feature for messaging (chat, etc.) using PGVector
'''

from uuid import UUID
from datetime import datetime, timezone
from typing import Iterable
import warnings

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
    ts
{limit_clause};
'''

SEMANTIC_QUERY_MESSAGE_TABLE = '''-- Find messages with closest semantic similarity
SELECT
    1 - (embedding <=> $1) AS cosine_similarity,
    ts,
    role,
    content,
    metadata
FROM
    {table_name}
WHERE
    history_key = $2
ORDER BY
    cosine_similarity DESC
LIMIT $3;
'''
# ------ SQL queries ---------------------------------------------------------------------------------------------------

class MessageDB(PGVectorHelper):
    ''' Specialize PGvectorHelper for messages, e.g. chatlogs '''
    async def create_table(self):
        async with (await self.connection_pool()).acquire() as conn:
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

            metadata (dict[str, str], optional): additional metadata of the message

        Returns:
            generator which yields the rows os the query results ass attributable dicts
        '''
        if not timestamp:
            timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)

        # Get the embedding of the content as a PGvector compatible list
        content_embedding = self._embedding_model.encode(content)

        async with (await self.connection_pool()).acquire() as conn:
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
        async with (await self.connection_pool()).acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    INSERT_MESSAGE.format(table_name=self.table_name),
                    (
                        (hk, role, text, self._embedding_model.encode(text), ts, metadata)
                        for hk, role, text, ts, metadata in content_list
                    )
                )

    async def clear(
            self,
            history_key: UUID
    ) -> None:
        '''
        Remove all entries in a message history entries

        Args:
            history_key (str): history key (unique identifier) to match
        '''
        async with (await self.connection_pool()).acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    CLEAR_MESSAGE.format(
                        table_name=self.table_name,
                    ),
                    history_key
                )

    # XXX: Change to a generator
    async def get_chatlog(
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
        qparams = [history_key]
        async with (await self.connection_pool()).acquire() as conn:
            if not isinstance(history_key, UUID):
                history_key = UUID(history_key)
                # msg = f'history_key must be a UUID, not {type(history_key)} ({history_key}))'
                # raise TypeError(msg)
            if not isinstance(since, datetime) and since is not None:
                msg = 'since must be a datetime or None'
                raise TypeError(msg)
            if since:
                # Really don't need the ${len(qparams) + 1} trick here, but used for consistency
                since_clause = f' AND ts > ${len(qparams) + 1}'
                qparams.append(since)
            else:
                since_clause = ''
            if limit:
                limit_clause = f'LIMIT ${len(qparams) + 1}'
                qparams.append(limit)
            else:
                limit_clause = ''
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

        # async for record in message_records:
        #     yield attr_dict({
        #         'ts': record['ts'],
        #         'role': record['role'],
        #         'content': record['content'],
        #         'metadata': record['metadata']
        #     })
    
    async def get_table(self, history_key):
        # Deprecated
        warnings.warn('pgvector_message.get_table() is deprecated. Use get_chatlog().', DeprecationWarning)
        return list(await self.get_chatlog(history_key))

    async def search(
            self,
            history_key: UUID,
            text: str,
            since: datetime | None = None,
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
        if not isinstance(limit, int):
            raise TypeError('limit must be an integer')

        # Get the embedding of the query string as a PGvector compatible list
        query_embedding = list(self._embedding_model.encode(text))

        # Search the table
        async with (await self.connection_pool()).acquire() as conn:
            records = await conn.fetch(
                SEMANTIC_QUERY_MESSAGE_TABLE.format(
                    table_name=self.table_name
                ),
                query_embedding,
                history_key,
                limit
            )

        search_results = [
            {
                'ts': record['ts'],
                'role': record['role'],
                'content': record['content'],
                'metadata': record['metadata'],
                'cosine_similarity': record['cosine_similarity']
            }
            for record in records
        ]

        return process_search_response(search_results)
