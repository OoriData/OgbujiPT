# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.pgvector

'''
Vector databases embeddings using PGVector
'''

# import warnings
# import itertools
import json
from typing   import Iterable
from uuid     import UUID
from datetime import datetime, timezone

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
# Table names are checked to be legit sequel table names, and embed_dimension is assured to be an integer
CREATE_VECTOR_EXTENSION = 'CREATE EXTENSION IF NOT EXISTS vector;'

CHECK_TABLE_EXISTS = '''-- Check if a table exists
SELECT EXISTS (
    SELECT FROM pg_tables
    WHERE tablename = $1
);
'''

# Generic SQL template for creating a table to hold embedded documents
CREATE_DOC_TABLE = '''-- Create a table to hold embedded documents
CREATE TABLE IF NOT EXISTS {table_name} (
    id BIGSERIAL PRIMARY KEY,
    embedding VECTOR({embed_dimension}),  -- embedding vectors (array dimension)
    content TEXT NOT NULL,                -- text content of the chunk
    title TEXT,                           -- title of file
    page_numbers INTEGER[],               -- page number of the document that the chunk is found in
    tags TEXT[]                           -- tags associated with the chunk
);
'''

INSERT_DOCS = '''-- Insert a document into a table
INSERT INTO {table_name} (
    embedding,
    content,
    title,
    page_numbers,
    tags
) VALUES ($1, $2, $3, $4, $5);
'''

QUERY_DOC_TABLE = '''-- Semantic search a document
SELECT
    1 - (embedding <=> '{query_embedding}') AS cosine_similarity,
    title,
    content,
    page_numbers,
    tags
FROM
    {table_name}
{where_clauses}\
ORDER BY
    cosine_similarity DESC
LIMIT {limit};
'''

TITLE_WHERE_CLAUSE = 'title = {query_title}  -- Equals operator \n'

PAGE_NUMBERS_WHERE_CLAUSE = 'page_numbers && {query_page_numbers}  -- Overlap operator \n'

TAGS_WHERE_CLAUSE_CONJ = 'tags @> {query_tags}  -- Contains operator \n'
TAGS_WHERE_CLAUSE_DISJ = 'tags && {query_tags}  -- Overlap operator \n'
# ----------------------------------------------------------------------------------------------------------------------
# Generic SQL template for creating a table to hold individual messages from a chatlog and their metadata
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
    1 - (embedding <=> '{query_embedding}') AS cosine_similarity,
    ts,
    role,
    content,
    metadata_JSON
FROM
    {table_name}
WHERE
    history_key = $1
ORDER BY
    cosine_similarity DESC
LIMIT $2;
'''
# ======================================================================================================================

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


class PGVectorHelper:
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
    async def from_conn_params(
            cls,
            embedding_model,
            table_name,
            user, 
            password,
            db_name,
            host,
            port,
            **conn_params
    ) -> 'PGVectorHelper':
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
            raise e
        return await cls.from_connection(embedding_model, table_name, conn)

    @classmethod
    async def from_connection(cls, embedding_model, table_name, conn) -> 'PGVectorHelper':
        '''
        Create a PGvector helper from connection parameters

        For details on accepted parameters, See the `pgvector_connection` docstring
            (e.g. run `help(pgvector_connection)`)
        '''
        # Ensure the vector extension is installed
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
        await register_vector(conn)

        await conn.set_type_codec(  # Register a codec for JSON
            'JSON',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )

        # print('PGvector extension created and loaded.')
        return cls(embedding_model, table_name, conn)

    # Hmm. Just called count in the qdrant version
    async def count_items(self) -> int:
        '''
        Count the number of documents in the table

        Returns:
            int: number of documents in the table
        '''
        # Count the number of documents in the table
        count = await self.conn.fetchval(f'SELECT COUNT(*) FROM {self.table_name}')
        return count
    
    async def table_exists(self) -> bool:
        '''
        Check if the table exists

        Returns:
            bool: True if the table exists, False otherwise
        '''
        # Check if the table exists
        table_exists = await self.conn.fetchval(
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
        await self.conn.execute(f'DROP TABLE IF EXISTS {self.table_name};')


class DocDB(PGVectorHelper):
    ''' Specialize PGvectorHelper for documents '''
    async def create_table(self) -> None:
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
    
    async def insert(
            self,
            content: str,
            title: str = 'NULL',
            page_numbers: list[int] = [],
            tags: list[str] = []
    ) -> None:
        '''
        Update a table with one embedded document

        Args:
            content (str): text content of the document

            title (str, optional): title of the document

            page_numbers (list[int], optional): page number of the document that the chunk is found in

            tags (list[str], optional): tags associated with the document
        '''
        # Get the embedding of the content as a PGvector compatible list
        content_embedding = self._embedding_model.encode(content)

        await self.conn.execute(
            INSERT_DOCS.format(table_name=self.table_name),
            content_embedding.tolist(),
            content,
            title,
            page_numbers,
            tags
        )

    async def insert_many(
            self,
            content_list: Iterable[tuple[str, str | None, list[int], list[str]]]
    ) -> None:
        '''
        Update a table with one or more embedded documents

        Semantically equivalent to multiple insert_doc calls, but uses executemany for efficiency

        Args:
            content_list: List of tuples, each of the form: (content, title, page_numbers, tags)
        '''
        await self.conn.executemany(
            INSERT_DOCS.format(table_name=self.table_name),
            (
                (self._embedding_model.encode(content), content, title, page_numbers, tags)
                for content, title, page_numbers, tags in content_list
            )
        )

    async def search(
            self,
            query_string: str,
            query_title: str | None = None,
            query_page_numbers: list[int] | None = None,
            query_tags: list[str] | None = None,
            limit: int = 1,
            conjunctive: bool = True
    ) -> list[asyncpg.Record]:
        '''
        Similarity search documents using a query string

        Args:
            query_string (str): string to compare against items in the table.
                This will be a vector/fuzzy/nearest-neighbor type search.

            query_title (str, optional): title of the document to compare against items in the table.

            query_page_numbers (list[int], optional): target page number in the document for query string comparison.

            query_tags (list[str], optional): tags associated with the document to compare against items in the table.
                Each individual tag must match exactly, but see the conjunctive param
                for how multiple tags are interpreted.

            limit (int, optional): maximum number of results to return (useful for top-k query)

            conjunctive (bool, optional): whether to use conjunctive (AND) or disjunctive (OR) matching
                in the case of multiple tags. Defaults to True.
        Returns:
            list[asyncpg.Record]: list of search results
                (asyncpg.Record objects are similar to dicts, but allow for attribute-style access)
        '''
        if not isinstance(limit, int):
            raise TypeError('limit must be an integer')

        # Get the embedding of the query string as a PGvector compatible list
        query_embedding = list(self._embedding_model.encode(query_string))

        tags_where_clause = TAGS_WHERE_CLAUSE_CONJ if conjunctive else TAGS_WHERE_CLAUSE_DISJ

        # Build where clauses
        if (query_title is None) and (query_page_numbers is None) and (query_tags is None):
            # No where clauses, so don't bother with the WHERE keyword
            where_clauses = ''
            query_args = []
        else:  # construct where clauses
            param_count = 0
            clauses = []
            query_args = []
            if query_title is not None:
                param_count += 1
                query_args.append(query_title)
                clauses.append(TITLE_WHERE_CLAUSE.format(query_title=f'${param_count}'))
            if query_page_numbers is not None:
                param_count += 1
                query_args.append(query_page_numbers)
                clauses.append(PAGE_NUMBERS_WHERE_CLAUSE.format(query_page_numbers=f'${param_count}'))
            if query_tags is not None:
                param_count += 1
                query_args.append(query_tags)
                clauses.append(tags_where_clause.format(query_tags=f'${param_count}'))
            clauses = 'AND\n'.join(clauses)  # TODO: move this into the fstring below after py3.12
            where_clauses = f'WHERE\n{clauses}'

        # Execute the search via SQL
        search_results = await self.conn.fetch(
            QUERY_DOC_TABLE.format(
                table_name=self.table_name,
                query_embedding=query_embedding,
                where_clauses=where_clauses,
                limit=limit
            ),
            *query_args
        )
        return search_results
    

class MessageDB(PGVectorHelper):
    ''' Specialize PGvectorHelper for chatlogs '''
    async def create_table(self):
        '''
        Create the table to hold chatlogs
        '''
        # Check that the connection is still alive
        # if self.conn.is_closed():
        #     raise ConnectionError('Connection to database is closed')

        # Create the table
        await self.conn.execute(
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
        '''
        if not timestamp:
            timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)

        role_int = role_to_int(role)  # Convert from string roles to integer roles

        # Get the embedding of the content as a PGvector compatible list
        content_embedding = self._embedding_model.encode(content)

        await self.conn.execute(
            INSERT_CHATLOG.format(table_name=self.table_name),
            timestamp,
            history_key,
            role_int,
            content,
            content_embedding.tolist(),
            metadata
        )   
    
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
        # Get the chatlog
        chatlog_records = await self.conn.fetch(
            RETURN_CHATLOG_BY_HISTORY_KEY.format(
                table_name=self.table_name,
            ),
            history_key
        )

        chatlog = [
            {
                'ts': record['ts'],
                'role': int_to_role(record['role']),
                'content': record['content'],
                'metadata': record['metadata_json']
            }
            for record in chatlog_records
        ]

        return chatlog
    
    async def search(
            self,
            history_key: UUID,
            query_string: str,
            limit: int = 1
    ) -> list[asyncpg.Record]:
        '''
        Similarity search documents using a query string

        Args:
            query_string (str): string to compare against items in the table

            k (int, optional): maximum number of results to return (useful for top-k query)
        Returns:
            list[asyncpg.Record]: list of search results
                (asyncpg.Record objects are similar to dicts, but allow for attribute-style access)
        '''
        if not isinstance(limit, int):
            raise TypeError('limit must be an integer')

        # Get the embedding of the query string as a PGvector compatible list
        query_embedding = list(self._embedding_model.encode(query_string))

        # Search the table
        records = await self.conn.fetch(
            SEMANTIC_QUERY_CHATLOG_TABLE.format(
                table_name=self.table_name,
                query_embedding=query_embedding
            ),
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

        return search_results
