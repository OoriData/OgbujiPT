# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.pgvector_data_doc
'''
Vector databases embeddings using PGVector
'''
import json
from typing import Iterable, Callable, List, Sequence

from ogbujipt.embedding.pgvector import (PGVectorHelper, asyncpg, process_search_response)

__all__ = ['DataDB']

# ------ SQL queries ---------------------------------------------------------------------------------------------------
# PG only supports proper query arguments (e.g. $1, $2, etc.) for values, not for table or column names
# Table names are checked to be legit sequel table names, and embed_dimension is assured to be an integer

CREATE_DATA_TABLE = '''-- Create a table to hold embedded documents or data
{set_schema}CREATE TABLE IF NOT EXISTS {table_name} (
    id BIGSERIAL PRIMARY KEY,
    embedding {type}({embed_dimension}),    -- embedding vectors (array dimension)
    content TEXT NOT NULL,                  -- text content of the chunk
    metadata jsonb                           -- additional metadata of the chunk
)
'''

# itype: vector, halfvec (0.7.0 & up), bit, sparsevec (0.7.0+)
# func: l2, ip, cosine, l1, hamming (0.7.0+, with bit type only), jaccard (0.7.0+, with bit type only)
CREATE_DATA_INDEX_HNSW = '''-- Create a table to hold a full precision HNSW index
CREATE INDEX
    IF NOT EXISTS {table_tail}_{itype}_{func} ON {table_name}
    USING hnsw ((embedding::{type}({embed_dimension})) {itype}_{func}_ops)
    WITH (m = {max_conn}, ef_construction = {ef_construction});
'''

INSERT_DATA = '''-- Insert a document into a table
INSERT INTO {table_name} (
    embedding,
    content,
    metadata) VALUES ($1, $2, $3);
'''

QUERY_DATA_TABLE = '''-- Semantic search a document
SELECT * FROM (  -- Subquery to calculate cosine similarity, required to use the alias in the WHERE clause
    SELECT
        1 - (embedding <=> $1) AS cosine_similarity,
        content,
        metadata
    FROM
        {table_name}
) subquery
{where_clauses}
ORDER BY
    cosine_similarity DESC
{limit_clause};
'''

# TITLE_WHERE_CLAUSE = 'title = {query_title}  -- Equals operator\n'

# PAGE_NUMBERS_WHERE_CLAUSE = 'page_numbers && {query_page_numbers}  -- Overlap operator \n'

THRESHOLD_WHERE_CLAUSE = 'cosine_similarity >= {query_threshold}\n'
# ------ SQL queries ---------------------------------------------------------------------------------------------------


class DataDB(PGVectorHelper):
    ''' Specialize PGvectorHelper for data (snippets) '''
    async def create_table(self) -> None:
        '''
        Create the table to hold embedded documents
        '''
        # set_schema = f'SET SCHEMA \'{self.schema}\';\n' if self.schema else ''
        set_schema = ''
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    CREATE_DATA_TABLE.format(
                        set_schema=set_schema,
                        type=self.vtype,
                        table_name=self.table_name,
                        embed_dimension=self._embed_dimension)
                    )
                for it in self.itypes:
                    for f in self.ifuncs:
                        await conn.execute(CREATE_DATA_INDEX_HNSW.format(
                            table_name=self.table_name,
                            table_tail=self.table_name.split('.')[-1],
                            type=self.vtype,
                            embed_dimension=self._embed_dimension,
                            itype=it,
                            func=f,
                            max_conn=self.i_max_conn,
                            ef_construction=self.ef_construction)
                        )

    async def insert(
            self,
            content: str,
            metadata: dict | None = None
    ) -> None:
        '''
        Update a table with one embedded document

        Args:
            content (str): text content of the document

            metadata (dict, optional): additional metadata of the chunk
        '''
        if self.stringify_json:
            metadata = json.dumps(metadata)
        # Get the embedding of the content as a PGvector compatible list
        content_embedding = self._embedding_model.encode(content)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    INSERT_DATA.format(table_name=self.table_name),
                    content_embedding,
                    content,
                    metadata
                )

    async def insert_many(
            self,
            content_list: Iterable[tuple[str, dict]]
    ) -> None:
        '''
        Update a table with one or (presumably) more embedded documents

        Semantically equivalent to multiple insert calls, but uses executemany for efficiency

        Args:
            content_list: List of tuples, each of the form: (content, metadata), where
                `content` is a string and
                `metadata` is a dictionary
        '''
        def handle_metadata(md):
            return json.dumps(md) if self.stringify_json else md

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    INSERT_DATA.format(table_name=self.table_name),
                    (
                        # Does this need to be .tolist()?
                        (self._embedding_model.encode(content), content, handle_metadata(metadata))
                        for content, metadata in content_list
                    )
                )

    async def search(
            self,
            text: str,
            threshold: float | None = None,
            meta_filter: Callable | List[Callable] | None = None,
            limit: int = 0
    ) -> list[asyncpg.Record]:
        '''
        Similarity search documents using a query string

        Args:
            text (str): string to compare against items in the table.
                This will be a vector/fuzzy/nearest-neighbor type search.

            threshold: minimum vector similarity to return

            meta_filter: specifies further restrictions on results, via one or more functions setting up SQL conditions
                for the metadata field. Can be a callable or list of callables, which will be ANDed together
                e.g.
                ```
                from ogbujipt.embedding.pgvector import match_exact
                page_one_filt = match_exact('page', 1)
                await DB.search(text='Hello!', meta_filter=page_one_filt)
                ```
                Provided filter helpers such as match_exact provide some SQL injection safety. If you roll your own,
                be careful of injection.
            
            limit (int, optional): maximum number of results to return (useful for top-k query)
                Default is no limit

        Returns:
            generator which yields the rows os the query results ass attributable dicts
        '''
        if threshold is not None:
            if not isinstance(threshold, float) or (threshold < 0) or (threshold > 1):
                raise TypeError('threshold must be a float between 0.0 and 1.0')
        if not isinstance(limit, int):
            raise TypeError('limit must be an integer')  # Guard against injection
        meta_filter = meta_filter or ()
        if not isinstance(meta_filter, Sequence):
            meta_filter = (meta_filter,)

        # Get the embedding of the query string as a PGvector compatible list
        query_embedding = list(self._embedding_model.encode(text))

        # Build where clauses
        query_args = [query_embedding]
        where_clauses = []
        if threshold is not None:
            query_args.append(threshold)
            where_clauses.append(THRESHOLD_WHERE_CLAUSE.format(query_threshold=f'${len(query_args)}'))

        for mf in meta_filter:
            assert callable(mf), 'All meta_filter items must be callable'
            clause, pval = mf()
            query_args.append(pval)
            where_clauses.append(clause.format(len(query_args)))

        where_clauses_str = 'WHERE\n' + 'AND\n'.join(where_clauses) if where_clauses else ''

        if limit:
            limit_clause = f'LIMIT {limit}\n'
        else:
            limit_clause = ''

        # print(QUERY_DATA_TABLE.format(table_name=self.table_name, where_clauses=where_clauses_str,
        #                             limit_clause=limit_clause,
        #     ))
        # print(query_args)

        # Execute the search via SQL
        async with self.pool.acquire() as conn:
            # Uncomment to debug
            # from asyncpg import utils
            # print(await utils._mogrify(
            #     conn,
            #     QUERY_DATA_TABLE.format(table_name=self.table_name, where_clauses=where_clauses_str,
            #                             limit_clause=limit_clause,
            #     ),
            #     query_args
            # ))
            search_results = await conn.fetch(
                QUERY_DATA_TABLE.format(table_name=self.table_name, where_clauses=where_clauses_str,
                                        limit_clause=limit_clause,
                ),
                *query_args
            )
        if self.stringify_json:
            search_results = ({
                'cosine_similarity': r['cosine_similarity'],
                'content': r['content'],
                'metadata': json.loads(r['metadata']),
                } for r in search_results)

        return process_search_response(search_results)
