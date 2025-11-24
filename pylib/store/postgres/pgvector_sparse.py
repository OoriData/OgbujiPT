# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.store.postgres.pgvector_sparse
'''
Sparse vector storage using PGVector sparsevec type (pgvector 0.7.0+).

Sparse vectors are efficient for high-dimensional vectors with mostly zero values,
such as:
- BM25/TF-IDF term vectors
- One-hot encoded categorical features
- Vocabulary-based embeddings

The sparsevec type stores only non-zero elements, saving storage and computation.

Philosophy: Extend existing pgvector patterns with minimal new abstractions.
'''

import json
from typing import AsyncIterator, Optional, Any

import structlog

from ogbujipt.store.postgres.pgvector import PGVectorHelper, asyncpg, process_search_response
from ogbujipt.memory.base import SearchResult


logger = structlog.get_logger()


__all__ = ['SparseDB']


# ------ SQL queries ---------------------------------------------------------------------------------------------------

CREATE_SPARSE_TABLE = '''-- Create a table to hold sparse vectors
{set_schema}CREATE TABLE IF NOT EXISTS {table_name} (
    id BIGSERIAL PRIMARY KEY,
    embedding sparsevec({embed_dimension}),  -- sparse vector (stores only non-zero elements)
    content TEXT NOT NULL,                    -- text content
    metadata jsonb                            -- additional metadata
)
'''

# sparsevec supports: l2, cosine, l1 distance functions
# HNSW indexing for sparse vectors
CREATE_SPARSE_INDEX_HNSW = '''-- Create HNSW index for sparse vectors
CREATE INDEX
    IF NOT EXISTS {table_tail}_sparsevec_{func} ON {table_name}
    USING hnsw (embedding sparsevec_{func}_ops)
    WITH (m = {max_conn}, ef_construction = {ef_construction});
'''

INSERT_SPARSE = '''-- Insert a sparse vector
INSERT INTO {table_name} (
    embedding,
    content,
    metadata) VALUES ($1, $2, $3)
RETURNING id;
'''

QUERY_SPARSE_TABLE = '''-- Search sparse vectors
SELECT * FROM (
    SELECT
        1 - (embedding <=> $1) AS similarity,  -- cosine similarity for sparsevec
        content,
        metadata,
        id
    FROM
        {table_name}
) subquery
{where_clauses}
ORDER BY
    similarity DESC
{limit_clause};
'''

THRESHOLD_WHERE_CLAUSE = 'similarity >= {query_threshold}\n'

# ------ SQL queries ---------------------------------------------------------------------------------------------------


class SparseDB(PGVectorHelper):
    '''
    Specialized PGVectorHelper for sparse vectors (sparsevec type).

    Designed for high-dimensional sparse embeddings like BM25 term vectors.
    Only non-zero elements are stored, making it efficient for large vocabularies.

    Example:
        >>> from ogbujipt.store.postgres import SparseDB
        >>> from sentence_transformers import SentenceTransformer
        >>>
        >>> # For demo - in production use actual sparse encoder
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>>
        >>> db = await SparseDB.from_conn_params(
        ...     embedding_model=model,
        ...     table_name='sparse_docs',
        ...     host='localhost',
        ...     port=5432,
        ...     db_name='mydb',
        ...     user='user',
        ...     password='pass',
        ...     vocab_size=10000,  # Sparse vector dimension
        ...     itypes=['sparsevec'],  # Use sparse indexing
        ...     ifuncs=['cosine']
        ... )
        >>> await db.create_table()
    '''

    def __init__(
        self,
        embedding_model,
        table_name: str,
        pool,
        vocab_size: int,  # Dimension of sparse vectors
        stringify_json=False,
        sys_schema='pg_catalog',
        itypes=None,
        ifuncs=None,
        i_max_conn=16,
        ef_construction=64
    ):
        '''
        Initialize sparse vector database.

        Args:
            embedding_model: Model that produces sparse vectors
            table_name: PostgreSQL table name
            pool: asyncpg connection pool
            vocab_size: Dimension of sparse vectors (e.g., vocabulary size for BM25)
            stringify_json: Whether to stringify JSON metadata
            sys_schema: Schema for vector extension
            itypes: Index types (should include 'sparsevec')
            ifuncs: Index functions (cosine, l1, l2 supported for sparsevec)
            i_max_conn: HNSW max connections per layer
            ef_construction: HNSW construction parameter
        '''
        # Initialize parent with half_precision=False (not applicable to sparsevec)
        super().__init__(
            embedding_model=embedding_model,
            table_name=table_name,
            pool=pool,
            stringify_json=stringify_json,
            sys_schema=sys_schema,
            half_precision=False,  # Not used for sparse
            itypes=itypes or ['sparsevec'],
            ifuncs=ifuncs or ['cosine'],
            i_max_conn=i_max_conn,
            ef_construction=ef_construction
        )

        self.vocab_size = vocab_size
        # Override vtype for sparse vectors
        self.vtype = 'sparsevec'

    async def create_table(self) -> None:
        '''Create table for sparse vectors with appropriate indices.'''
        set_schema = ''
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Create main table
                await conn.execute(
                    CREATE_SPARSE_TABLE.format(
                        set_schema=set_schema,
                        table_name=self.table_name,
                        embed_dimension=self.vocab_size
                    )
                )

                # Create indices for each itype/ifunc combo
                # Only sparsevec indices are valid here
                for it in self.itypes:
                    if it != 'sparsevec':
                        logger.warning('sparse_invalid_itype',
                                     itype=it, msg='Only sparsevec supported for SparseDB')
                        continue

                    for f in self.ifuncs:
                        if f not in ('l2', 'cosine', 'l1'):
                            logger.warning('sparse_invalid_ifunc',
                                         ifunc=f, msg='Only l2/cosine/l1 supported for sparsevec')
                            continue

                        await conn.execute(
                            CREATE_SPARSE_INDEX_HNSW.format(
                                table_name=self.table_name,
                                table_tail=self.table_name.split('.')[-1],
                                func=f,
                                max_conn=self.i_max_conn,
                                ef_construction=self.ef_construction
                            )
                        )

        logger.info('sparse_table_created', table=self.table_name, vocab_size=self.vocab_size)

    async def insert(
        self,
        content: str,
        sparse_vector: dict[int, float] | list[tuple[int, float]],  # Sparse format: {idx: val} or [(idx, val), ...]
        metadata: Optional[dict] = None
    ) -> int:
        '''
        Insert a document with its sparse vector.

        Args:
            content: Text content
            sparse_vector: Sparse vector as dict {index: value} or list [(index, value), ...]
                Only non-zero elements need to be specified.
            metadata: Optional metadata dict

        Returns:
            Document ID (bigint)

        Example:
            >>> # Sparse vector with vocab_size=1000, only 3 non-zero terms
            >>> sparse_vec = {42: 0.5, 100: 0.8, 333: 0.3}
            >>> doc_id = await db.insert('Machine learning is great', sparse_vec)
        '''
        if self.stringify_json:
            metadata = json.dumps(metadata)

        # Convert sparse vector to pgvector format
        # sparsevec expects: '{index1:value1,index2:value2,...}/dimension'
        if isinstance(sparse_vector, dict):
            sparse_items = sparse_vector.items()
        else:
            sparse_items = sparse_vector

        # Sort by index for consistency
        sorted_items = sorted(sparse_items, key=lambda x: x[0])

        # Format: {1:0.5,5:0.8,10:0.3}/100
        sparse_str = '{' + ','.join(f'{idx}:{val}' for idx, val in sorted_items) + f'}}/{self.vocab_size}'

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                doc_id = await conn.fetchval(
                    INSERT_SPARSE.format(table_name=self.table_name),
                    sparse_str,
                    content,
                    metadata
                )

        logger.debug('sparse_insert', doc_id=doc_id, nnz=len(sorted_items))
        return doc_id

    async def search(
        self,
        query: str | dict[int, float] | list[tuple[int, float]],  # Text or sparse vector
        threshold: Optional[float] = None,
        limit: int = 5,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''
        Search using sparse vectors.

        Args:
            query: Either text (will be encoded) or sparse vector dict/list
            threshold: Minimum similarity threshold
            limit: Maximum results to return
            **kwargs: Additional options (for protocol compatibility)

        Yields:
            SearchResult objects

        Note: This implements the KBBackend protocol's search method.
        '''
        # Encode query if it's text
        if isinstance(query, str):
            if self._embedding_model is None:
                logger.error('sparse_no_encoder')
                raise ValueError('No embedding model provided, cannot encode text query')

            # Assume the model produces sparse vectors
            # In practice, you'd use a BM25 encoder or similar
            sparse_vector = self._embedding_model.encode(query)

            # If model returns dense vector, convert to sparse (naive approach)
            # For production, use proper sparse encoders
            if not isinstance(sparse_vector, (dict, list)):
                logger.warning('sparse_dense_encoder',
                             msg='Model returned dense vector, converting to sparse (may be inefficient)')
                # Simple thresholding: only keep values > 0.01
                sparse_vector = {i: v for i, v in enumerate(sparse_vector) if abs(v) > 0.01}
        else:
            sparse_vector = query

        # Convert to pgvector sparse format
        if isinstance(sparse_vector, dict):
            sparse_items = sparse_vector.items()
        else:
            sparse_items = sparse_vector

        sorted_items = sorted(sparse_items, key=lambda x: x[0])
        sparse_str = '{' + ','.join(f'{idx}:{val}' for idx, val in sorted_items) + f'}}/{self.vocab_size}'

        # Build SQL query
        query_args = [sparse_str]
        where_clauses = []

        if threshold is not None:
            if not isinstance(threshold, float) or not (0 <= threshold <= 1):
                raise TypeError('threshold must be a float between 0.0 and 1.0')
            query_args.append(threshold)
            where_clauses.append(THRESHOLD_WHERE_CLAUSE.format(query_threshold=f'${len(query_args)}'))

        where_clauses_str = 'WHERE\n' + 'AND\n'.join(where_clauses) if where_clauses else ''
        limit_clause = f'LIMIT {limit}' if limit else ''

        # Execute search
        async with self.pool.acquire() as conn:
            search_results = await conn.fetch(
                QUERY_SPARSE_TABLE.format(
                    table_name=self.table_name,
                    where_clauses=where_clauses_str,
                    limit_clause=limit_clause
                ),
                *query_args
            )

        # Convert to SearchResult objects
        for row in search_results:
            metadata = row['metadata']
            if self.stringify_json and metadata:
                metadata = json.loads(metadata)

            result = SearchResult(
                content=row['content'],
                score=float(row['similarity']),
                metadata=metadata or {},
                source=f'{self.table_name}_sparse'
            )

            yield result

        logger.debug('sparse_search_complete', returned=len(search_results))

    # Alias for protocol compatibility
    async def delete(self, item_id: int, **kwargs) -> bool:
        '''
        Delete a document by ID.

        Args:
            item_id: Document ID to delete
            **kwargs: Additional options (unused)

        Returns:
            True if deleted, False if not found
        '''
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f'DELETE FROM {self.table_name} WHERE id = $1',
                item_id
            )

        # result is like 'DELETE 1' or 'DELETE 0'
        deleted = int(result.split()[-1]) > 0
        logger.debug('sparse_delete', doc_id=item_id, deleted=deleted)
        return deleted
