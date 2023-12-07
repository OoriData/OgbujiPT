# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.pgvector_chat

'''
Vector databases embeddings using PGVector
'''

from uuid     import UUID
from datetime import datetime, timezone

from ogbujipt.config import attr_dict
from ogbujipt.embedding.pgvector import PGVectorHelper, asyncpg, process_search_response

__all__ = ['MessageDB']

# ------ SQL queries ---------------------------------------------------------------------------------------------------
# PG only supports proper query arguments (e.g. $1, $2, etc.) for values, not for table or column names
# Table names are checked to be legit sequel table names, and embed_dimension is assured to be an integer

CREATE_CHATLOG_TABLE = '''-- Create a table to hold individual messages from a chatlog and their metadata
CREATE TABLE IF NOT EXISTS {table_name} (
    ts TIMESTAMP WITH TIME ZONE PRIMARY KEY,  -- timestamp of the message
    history_key UUID,                         -- history key (unique identifier) of the chatlog this message belongs to
    role INT,                                 -- role of the message
    content TEXT NOT NULL,                    -- text content of the message
    embedding VECTOR({embed_dimension}),      -- embedding vectors (array dimension)
    metadata_JSON JSON                        -- additional metadata of the message
);
'''

INSERT_CHATLOG = '''-- Insert a message into a chatlog
INSERT INTO {table_name} (
    ts,
    history_key,
    role,
    content,
    embedding,
    metadata_JSON
) VALUES ($1, $2, $3, $4, $5, $6);
'''

RETURN_CHATLOG_BY_HISTORY_KEY = '''-- Get entire chatlog of a history key
SELECT
    ts,
    role,
    content,
    metadata_JSON
FROM
    {table_name}
WHERE
    history_key = $1
ORDER BY
    ts;
'''

SEMANTIC_QUERY_CHATLOG_TABLE = '''-- Semantic search a chatlog
SELECT
    1 - (embedding <=> $1) AS cosine_similarity,
    ts,
    role,
    content,
    metadata_JSON
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
    ''' Specialize PGvectorHelper for chatlogs '''
    async def create_table(self):
        '''
        Create the table to hold chatlogs
        '''
        async with (await self.connection_pool()).acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    CREATE_CHATLOG_TABLE.format(
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
                    INSERT_CHATLOG.format(table_name=self.table_name),
                    timestamp,
                    history_key,
                    role_int,
                    content,
                    content_embedding.tolist(),
                    metadata
                )   
    
    # XXX: Change to a generator
    async def get_table(
            self,
            history_key: UUID
    ) -> list[asyncpg.Record]:
        '''
        Get the entire chatlog of a history key

        Args:
            history_key (str): history key of the chatlog
        Returns:
            list[asyncpg.Record]: list of chatlog
                (asyncpg.Record objects are similar to dicts, but allow for attribute-style access)
        '''
        async with (await self.connection_pool()).acquire() as conn:
            chatlog_records = await conn.fetch(
                RETURN_CHATLOG_BY_HISTORY_KEY.format(
                    table_name=self.table_name,
                ),
                history_key
            )

        chatlog = [
            attr_dict({
                'ts': record['ts'],
                'role': int_to_role(record['role']),
                'content': record['content'],
                'metadata': record['metadata_json']
            })
            for record in chatlog_records
        ]

        return chatlog
    
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
                SEMANTIC_QUERY_CHATLOG_TABLE.format(
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
                'metadata': record['metadata_json'],
                'cosine_similarity': record['cosine_similarity']
            }
            for record in records
        ]

        return process_search_response(search_results)
