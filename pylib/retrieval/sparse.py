# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.retrieval.sparse
'''
Sparse retrieval using BM25 (Best Matching 25) algorithm.

BM25 is a probabilistic retrieval function that ranks documents based on
query term frequency and document length normalization. Ideal for keyword-based
searches and as a complement to dense vector retrieval in hybrid systems.

Philosophy: Minimal abstractions. This is a search strategy that works with
any KBBackend that provides text content.
'''

from typing import AsyncIterator

from rank_bm25 import BM25Okapi
import structlog

from ogbujipt.memory.base import SearchResult, KBBackend  # , SearchStrategy


logger = structlog.get_logger()


class BM25Search:
    '''
    BM25 sparse retrieval strategy.

    Uses BM25Okapi algorithm to rank documents based on term frequency
    and inverse document frequency with document length normalization.

    Example:
        >>> from ogbujipt.retrieval.sparse import BM25Search
        >>> from ogbujipt.store.postgres import DataDB
        >>>
        >>> # Initialize with your backends
        >>> search = BM25Search()
        >>> results = search.execute('machine learning', backends=[db], limit=5)
        >>> async for result in results:
        ...     print(f'{result.score:.3f}: {result.content[:50]}...')

    Note: BM25 requires loading all documents into memory for indexing.
    For very large corpora (>100k docs), consider using a backend with
    built-in BM25 support or implementing incremental indexing.
    '''

    def __init__(
        self,
        k1: float = 1.5,  # Term frequency saturation parameter
        b: float = 0.75,  # Length normalization parameter
        epsilon: float = 0.25,  # IDF floor (prevents negative IDF values)
    ):
        '''
        Initialize BM25 search strategy.

        Args:
            k1: Controls term frequency saturation (default 1.5).
                Higher = more weight on term frequency. Range: [1.2, 2.0] typical.
            b: Controls document length normalization (default 0.75).
                0 = no normalization, 1 = full normalization. Range: [0, 1].
            epsilon: Floor for IDF values (default 0.25).
                Prevents negative IDF scores for very common terms.
        '''
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._index = None  # BM25 index, built on first search
        self._documents = []  # Document contents for retrieval
        self._metadata = []  # Metadata for each document
        self._source_map = []  # Which backend each doc came from

    async def _build_index(self, backends: list[KBBackend]) -> None:
        '''
        Build BM25 index from all documents in the backends.

        This loads all documents into memory. For large corpora, consider
        implementing a caching strategy or using a backend with native BM25.

        Args:
            backends: List of KB backends to index
        '''
        logger.info('bm25_index_building', backend_count=len(backends))

        self._documents = []
        self._metadata = []
        self._source_map = []

        # Fetch all documents from all backends
        # We use a large limit and no threshold to get everything
        for backend in backends:
            backend_name = backend.__class__.__name__
            doc_count = 0

            try:
                # Query with empty string to get all documents (backend-specific behavior)
                # For backends that don't support this, they should return everything
                async for result in backend.search(query='', limit=999999, threshold=None):
                    self._documents.append(result.content)
                    self._metadata.append(result.metadata)
                    self._source_map.append(result.source)
                    doc_count += 1
            except Exception as e:
                logger.warning('bm25_backend_indexing_failed',
                              backend=backend_name, error=str(e))
                continue

            logger.debug('bm25_backend_indexed', backend=backend_name, doc_count=doc_count)

        # Tokenize documents (simple whitespace + lowercase)
        # For production, consider more sophisticated tokenization
        tokenized_corpus = [doc.lower().split() for doc in self._documents]

        # Build BM25 index
        self._index = BM25Okapi(
            tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )

        logger.info('bm25_index_built', total_docs=len(self._documents))

    async def execute(
        self,
        query: str,
        backends: list[KBBackend],
        limit: int = 5,
        threshold: float | None = None,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''
        Execute BM25 search across backends.

        Args:
            query: Search query string
            backends: List of KB backends to search
            limit: Maximum number of results to return
            threshold: Minimum BM25 score threshold (optional)
            **kwargs: Additional options (unused, for protocol compatibility)

        Yields:
            SearchResult objects sorted by BM25 score (highest first)

        Note: The index is built on first search and cached. To rebuild,
        create a new BM25Search instance.
        '''
        if not backends:
            logger.warning('bm25_no_backends')
            return

        # Build index if not already built
        if self._index is None:
            await self._build_index(backends)

        if not self._documents:
            logger.info('bm25_no_documents')
            return

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores for all documents
        scores = self._index.get_scores(tokenized_query)

        # Create (score, index) pairs and sort by score descending
        scored_docs = [(score, idx) for idx, score in enumerate(scores)]
        scored_docs.sort(reverse=True, key=lambda x: x[0])

        # Normalize scores to [0, 1] range for consistency with other strategies
        # BM25 scores are unbounded, so we use a simple normalization
        max_score = scored_docs[0][0] if scored_docs and scored_docs[0][0] > 0 else 1.0

        # Yield results up to limit, applying threshold if specified
        returned = 0
        for score, idx in scored_docs:
            if returned >= limit:
                break

            normalized_score = score / max_score if max_score > 0 else 0.0

            # Apply threshold if specified
            if threshold is not None and normalized_score < threshold:
                continue

            result = SearchResult(
                content=self._documents[idx],
                score=normalized_score,
                metadata=self._metadata[idx],
                source=f'{self._source_map[idx]}_bm25'
            )

            yield result
            returned += 1

        logger.debug('bm25_search_complete', query_len=len(tokenized_query),
                    total_scored=len(scored_docs), returned=returned)


__all__ = ['BM25Search']
