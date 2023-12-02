# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.pgvector_doc

'''
Vector databases embeddings using PGVector
'''

from typing   import Iterable

from ogbujipt.embedding.pgvector import PGVectorHelper, asyncpg

__all__ = ['DocDB']


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
SELECT * FROM
-- Subquery to calculate cosine similarity, required to use the alias in the WHERE clause
(
    SELECT
        1 - (embedding <=> $1) AS cosine_similarity,
        title,
        content,
        page_numbers,
        tags
    FROM
        {table_name}
) subquery
{where_clauses}
ORDER BY
    cosine_similarity DESC
{limit_clause};
'''

TITLE_WHERE_CLAUSE = 'title = {query_title}  -- Equals operator \n'

PAGE_NUMBERS_WHERE_CLAUSE = 'page_numbers && {query_page_numbers}  -- Overlap operator \n'

TAGS_WHERE_CLAUSE_CONJ = 'tags @> {query_tags}  -- Contains operator \n'
TAGS_WHERE_CLAUSE_DISJ = 'tags && {query_tags}  -- Overlap operator \n'

THRESHOLD_WHERE_CLAUSE = '{query_threshold} >= cosine_similarity\n'


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
            threshold: float | None = None,
            limit: int = 0,
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
                Default is no limit

            conjunctive (bool, optional): whether to use conjunctive (AND) or disjunctive (OR) matching
                in the case of multiple tags. Defaults to True.
        Returns:
            list[asyncpg.Record]: list of search results
                (asyncpg.Record objects are similar to dicts, but allow for attribute-style access)
        '''
        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError('threshold must be a float')
            if (threshold < 0) or (threshold > 1):
                raise ValueError('threshold must be between 0 and 1')

        if not isinstance(limit, int):
            raise TypeError('limit must be an integer')  # Guard against injection

        # Get the embedding of the query string as a PGvector compatible list
        query_embedding = list(self._embedding_model.encode(query_string))

        tags_where_clause = TAGS_WHERE_CLAUSE_CONJ if conjunctive else TAGS_WHERE_CLAUSE_DISJ

        # Build where clauses
        if (query_title is None) and (query_page_numbers is None) and (query_tags is None) and (threshold is None):
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
            if query_tags is not None:
                param_count += 1
                query_args.append(query_tags)
                clauses.append(tags_where_clause.format(query_tags=f'${param_count}'))
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
        return search_results
