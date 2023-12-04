# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.pgvector_doc

'''
Vector databases embeddings using PGVector
'''

import warnings
from typing   import Iterable

from ogbujipt.embedding.pgvector import PGVectorHelper, asyncpg, process_search_response

__all__ = ['DocDB']


CREATE_TABLE_BASE = '''-- Create a table to hold embedded documents or data
CREATE TABLE IF NOT EXISTS {{table_name}} (
    id BIGSERIAL PRIMARY KEY,
    embedding VECTOR({{embed_dimension}}),  -- embedding vectors (array dimension)
    content TEXT NOT NULL,                -- text content of the chunk
    tags TEXT[]                           -- tags associated with the chunk
{extra_fields});
'''

CREATE_DOC_TABLE = CREATE_TABLE_BASE.format(extra_fields='''\
,    title TEXT,                           -- title of file
    page_numbers INTEGER[]               -- page number of the document that the chunk is found in
''')

CREATE_DATA_TABLE = CREATE_TABLE_BASE.format(extra_fields='')

INSERT_BASE = '''-- Insert a document into a table
INSERT INTO {{table_name}} (
    embedding,
    content,
    tags
    {extra_fields}) VALUES ($1, $2, $3{extra_vals});
'''

INSERT_DOCS = INSERT_BASE.format(extra_fields='''\
,    title,
    page_numbers
''', extra_vals=', $4, $5')

INSERT_DATA = INSERT_BASE.format(extra_fields='', extra_vals='')

QUERY_TABLE_BASE = '''-- Semantic search a document
SELECT * FROM
-- Subquery to calculate cosine similarity, required to use the alias in the WHERE clause
(
    SELECT
        1 - (embedding <=> $1) AS cosine_similarity,
        content,
        tags
        {extra_fields}
    FROM
        {{table_name}}
) subquery
{{where_clauses}}
ORDER BY
    cosine_similarity DESC
{{limit_clause}};
'''

QUERY_DOC_TABLE = QUERY_TABLE_BASE.format(extra_fields='''\
,        title,
        page_numbers
''')

QUERY_DATA_TABLE = QUERY_TABLE_BASE.format(extra_fields='')

TITLE_WHERE_CLAUSE = 'title = {query_title}  -- Equals operator \n'

PAGE_NUMBERS_WHERE_CLAUSE = 'page_numbers && {query_page_numbers}  -- Overlap operator \n'

TAGS_WHERE_CLAUSE_CONJ = 'tags @> {tags}  -- Contains operator \n'
TAGS_WHERE_CLAUSE_DISJ = 'tags && {tags}  -- Overlap operator \n'

THRESHOLD_WHERE_CLAUSE = '{query_threshold} >= cosine_similarity\n'


# XXX: Data vs doc DB can probably be modularized further, but this will do for now
class DataDB(PGVectorHelper):
    ''' Specialize PGvectorHelper for data (snippets) '''
    async def create_table(self) -> None:
        '''
        Create the table to hold embedded documents
        '''
        await self.conn.execute(
            CREATE_DATA_TABLE.format(
                table_name=self.table_name,
                embed_dimension=self._embed_dimension)
            )
    
    async def insert(
            self,
            content: str,
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
            INSERT_DATA.format(table_name=self.table_name),
            content_embedding.tolist(),
            content,
            tags
        )

    async def insert_many(
            self,
            content_list: Iterable[tuple[str, list[str]]]
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
                (self._embedding_model.encode(content), content, tags)
                for content, tags in content_list
            )
        )

    async def search(
            self,
            text: str,
            tags: list[str] | None = None,
            threshold: float | None = None,
            limit: int = 0,
            conjunctive: bool = True,
            query_tags: list[str] | None = None,
    ) -> list[asyncpg.Record]:
        '''
        Similarity search documents using a query string

        Args:
            text (str): string to compare against items in the table.
                This will be a vector/fuzzy/nearest-neighbor type search.

            tags (list[str], optional): tags associated with the document to compare against items in the table.
                Each individual tag must match exactly, but see the conjunctive param
                for how multiple tags are interpreted.

            limit (int, optional): maximum number of results to return (useful for top-k query)
                Default is no limit

            conjunctive (bool, optional): whether to use conjunctive (AND) or disjunctive (OR) matching
                in the case of multiple tags. Defaults to True.

        Returns:
            generator which yields the rows os the query results ass attributable dicts
        '''
        if query_tags is not None:
            warnings.warn('query_tags is deprecated. Use tags instead.', DeprecationWarning)
            tags = query_tags
        # else:
        #     if not isinstance(query_tags, list):
        #         raise TypeError('query_tags must be a list of strings')
        #     if not all(isinstance(tag, str) for tag in query_tags):
        #         raise TypeError('query_tags must be a list of strings')
        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError('threshold must be a float')
            if (threshold < 0) or (threshold > 1):
                raise ValueError('threshold must be between 0 and 1')

        if not isinstance(limit, int):
            raise TypeError('limit must be an integer')  # Guard against injection

        # Get the embedding of the query string as a PGvector compatible list
        query_embedding = list(self._embedding_model.encode(text))

        tags_where_clause = TAGS_WHERE_CLAUSE_CONJ if conjunctive else TAGS_WHERE_CLAUSE_DISJ

        # Build where clauses
        if (tags is None) and (threshold is None):
            # No where clauses, so don't bother with the WHERE keyword
            where_clauses = ''
            query_args = [query_embedding]
        else:  # construct where clauses
            param_count = 1
            clauses = []
            query_args = [query_embedding]
            if tags is not None:
                param_count += 1
                query_args.append(tags)
                clauses.append(tags_where_clause.format(tags=f'${param_count}'))
            if threshold is not None:
                param_count += 1
                query_args.append(threshold)
                clauses.append(THRESHOLD_WHERE_CLAUSE.format(query_threshold=f'${param_count}'))
            clauses = 'AND\n'.join(clauses)  # TODO: move this into the fstring below after py3.12
            where_clauses = f'WHERE\n{clauses}'

        if limit:
            limit_clause = f'LIMIT {limit}\n'
        else:
            limit_clause = ''

        # Execute the search via SQL
        search_results = await self.conn.fetch(
            QUERY_DATA_TABLE.format(
                table_name=self.table_name,
                where_clauses=where_clauses,
                limit_clause=limit_clause,
            ),
            *query_args
        )
        return process_search_response(search_results)


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
            tags,
            title,
            page_numbers
        )

    async def insert_many(
            self,
            content_list: Iterable[tuple[str, list[str], str | None, list[int]]]
    ) -> None:
        '''
        Update a table with one or more embedded documents

        Semantically equivalent to multiple insert_doc calls, but uses executemany for efficiency

        Args:
            content_list: List of tuples, each of the form: (content, tags, title, page_numbers)
        '''
        await self.conn.executemany(
            INSERT_DOCS.format(table_name=self.table_name),
            (
                (self._embedding_model.encode(content), content, tags, title, page_numbers)
                for content, tags, title, page_numbers in content_list
            )
        )

    async def search(
            self,
            text: str,
            query_title: str | None = None,
            query_page_numbers: list[int] | None = None,
            tags: list[str] | None = None,
            threshold: float | None = None,
            limit: int = 0,
            conjunctive: bool = True,
            query_tags: list[str] | None = None,
    ) -> list[asyncpg.Record]:
        '''
        Similarity search documents using a query string

        Args:
            text (str): string to compare against items in the table.
                This will be a vector/fuzzy/nearest-neighbor type search.

            query_title (str, optional): title of the document to compare against items in the table.

            query_page_numbers (list[int], optional): target page number in the document for query string comparison.

            tags (list[str], optional): tags associated with the document to compare against items in the table.
                Each individual tag must match exactly, but see the conjunctive param
                for how multiple tags are interpreted.

            limit (int, optional): maximum number of results to return (useful for top-k query)
                Default is no limit

            conjunctive (bool, optional): whether to use conjunctive (AND) or disjunctive (OR) matching
                in the case of multiple tags. Defaults to True.
        
        Returns:
            generator which yields the rows os the query results ass attributable dicts
        '''
        if query_tags is not None:
            warnings.warn('query_tags is deprecated. Use tags instead.', DeprecationWarning)
            tags = query_tags
        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError('threshold must be a float')
            if (threshold < 0) or (threshold > 1):
                raise ValueError('threshold must be between 0 and 1')

        if not isinstance(limit, int):
            raise TypeError('limit must be an integer')  # Guard against injection

        # Get the embedding of the query string as a PGvector compatible list
        query_embedding = list(self._embedding_model.encode(text))

        tags_where_clause = TAGS_WHERE_CLAUSE_CONJ if conjunctive else TAGS_WHERE_CLAUSE_DISJ

        # Build where clauses
        if (query_title is None) and (query_page_numbers is None) and (tags is None) and (threshold is None):
            # No where clauses, so don't bother with the WHERE keyword
            where_clauses = ''
            query_args = [query_embedding]
        else:  # construct where clauses
            param_count = 1
            clauses = []
            query_args = [query_embedding]
            if query_title is not None:
                param_count += 1
                query_args.append(query_title)
                clauses.append(TITLE_WHERE_CLAUSE.format(query_title=f'${param_count}'))
            if query_page_numbers is not None:
                param_count += 1
                query_args.append(query_page_numbers)
                clauses.append(PAGE_NUMBERS_WHERE_CLAUSE.format(query_page_numbers=f'${param_count}'))
            if tags is not None:
                param_count += 1
                query_args.append(tags)
                clauses.append(tags_where_clause.format(tags=f'${param_count}'))
            if threshold is not None:
                param_count += 1
                query_args.append(threshold)
                clauses.append(THRESHOLD_WHERE_CLAUSE.format(query_threshold=f'${param_count}'))
            clauses = 'AND\n'.join(clauses)  # TODO: move this into the fstring below after py3.12
            where_clauses = f'WHERE\n{clauses}'

        if limit:
            limit_clause = f'LIMIT {limit}\n'
        else:
            limit_clause = ''

        # Execute the search via SQL
        search_results = await self.conn.fetch(
            QUERY_DOC_TABLE.format(
                table_name=self.table_name,
                where_clauses=where_clauses,
                limit_clause=limit_clause,
            ),
            *query_args
        )
        return process_search_response(search_results)
