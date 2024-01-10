# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.pgvector_message

'''
Vector embeddings DB feature for messaging (chai, etc.) using PGVector
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
    ts TIMESTAMP WITH TIME ZONE PRIMARY KEY,  -- timestamp of the message
    history_key UUID,                         -- uunique identifier for contextual message history
    role INT,                                 -- role of the message (generally an ID associated with the sender)
    content TEXT NOT NULL,                    -- text content of the message
    embedding VECTOR({embed_dimension}),      -- embedding vectors (array dimension)
    metadata JSON                             -- additional metadata of the message
);
'''

INSERT_MESSAGE = '''-- Insert a message into a chatlog
INSERT INTO {table_name} (
    ts,
    history_key,
    role,
    content,
    embedding,
    metadata
) VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (ts) DO UPDATE SET  -- Update the content, embedding, and metadata of the message if it already exists
    content = EXCLUDED.content,
    embedding = EXCLUDED.embedding,
    metadata = EXCLUDED.metadata;
'''

CLEAR_MESSAGE = '''-- Deletes matching messages
DELETE FROM {table_name}
WHERE
    history_key = $1
'''

RETURN_MESSAGE_BY_HISTORY_KEY = '''-- Get entire chatlog by history key
SELECT
    ts,
    role,
    content,
    metadata
FROM
    {table_name}
WHERE
    history_key = $1
ORDER BY
    ts;
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

ROLE_INTS = {
    'system': 0,
    'user': 1,
    'assistant': 2,
    'tool': 3  # formerly 'function'
}

INT_ROLES = {v: k for k, v in ROLE_INTS.items()}


# Client code could avoid the function call overheads by just doing the dict lookups directly
def role_to_int(role):
    ''' Convert OpenAI style message role from strings to integers for faster DB ops '''
    return ROLE_INTS.get(role, -1)


def int_to_role(role_int):
    ''' Convert OpenAI style message role from integers to strings for faster DB ops '''
    return INT_ROLES.get(role_int, 'unknown')


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
            timestamp: datetime | None = None,
            metadata: dict | None = None
    ) -> None:
        '''
        Update a table with one embedded document

        Args:
            history_key (str): history key (unique identifier) of the chatlog this message belongs to

            role (str): role of the message (system, user, assistant, tool (formerly 'function'))

            content (str): text content of the message

            timestamp (datetime, optional): timestamp of the message, defaults to current time

            metadata (dict[str, str], optional): additional metadata of the message

        Returns:
            generator which yields the rows os the query results ass attributable dicts
        '''
        if not timestamp:
            timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)

        role_int = role_to_int(role)  # Convert from string roles to integer roles

        # Get the embedding of the content as a PGvector compatible list
        content_embedding = self._embedding_model.encode(content)

        async with (await self.connection_pool()).acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    INSERT_MESSAGE.format(table_name=self.table_name),
                    timestamp,
                    history_key,
                    role_int,
                    content,
                    content_embedding.tolist(),
                    metadata
                )   

    async def insert_many(
            self,
            content_list: Iterable[tuple[str, list[str]]]
    ) -> None:
        '''
        Update a table with one or more embedded messages

        Semantically equivalent to multiple insert calls, but uses executemany for efficiency

        Args:
            content_list: List of tuples, each of the form: (history_key, role, text, timestamp, metadata)
        '''
        async with (await self.connection_pool()).acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    INSERT_MESSAGE.format(table_name=self.table_name),
                    (
                        (ts, hk, role, text, self._embedding_model.encode(text), metadata)
                        for ts, hk, role, text, metadata in content_list
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
    async def get_table(
            self,
            history_key: UUID
    ) -> list[asyncpg.Record]:
        '''
        Retrieve all entries in a message history

        Args:
            history_key (str): history key (unique identifier) to match
        Returns:
            list[asyncpg.Record]: list of message entries
                (asyncpg.Record objects are similar to dicts, but allow for attribute-style access)
        '''
        async with (await self.connection_pool()).acquire() as conn:
            message_records = await conn.fetch(
                RETURN_MESSAGE_BY_HISTORY_KEY.format(
                    table_name=self.table_name,
                ),
                history_key
            )

        messages = [
            attr_dict({
                'ts': record['ts'],
                'role': int_to_role(record['role']),
                'content': record['content'],
                'metadata': record['metadata']
            })
            for record in message_records
        ]

        return messages
    
    async def search(
            self,
            history_key: UUID,
            text: str,
            limit: int = 1
    ) -> list[asyncpg.Record]:
        '''
        Similarity search documents using a query string

        Args:
            text (str): string to compare against items in the table

            k (int, optional): maximum number of results to return (useful for top-k query)
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
                'role': int_to_role(record['role']),
                'content': record['content'],
                'metadata': record['metadata'],
                'cosine_similarity': record['cosine_similarity']
            }
            for record in records
        ]

        return process_search_response(search_results)
