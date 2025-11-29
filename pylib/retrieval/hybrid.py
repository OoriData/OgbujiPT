# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.retrieval.hybrid
'''
Hybrid retrieval combining multiple search strategies using Reciprocal Rank Fusion (RRF).

RRF is a robust method for combining ranked lists from different retrieval systems.
It's parameter-free (except for the optional k constant) and has been shown to
consistently outperform individual retrieval methods.

Citation: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
"Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf

Philosophy: Minimal abstractions. Compose multiple search strategies without
complex weighting or tuning.
'''

from typing import AsyncIterator
from collections import defaultdict
import asyncio

import structlog

from ogbujipt.memory.base import SearchResult, SearchStrategy, KBBackend


logger = structlog.get_logger()


class HybridSearch:
    '''
    Hybrid search using Reciprocal Rank Fusion (RRF).

    Combines results from multiple search strategies (e.g., dense vector + sparse BM25)
    using RRF, which weights results by their rank rather than raw scores.

    RRF score for document d: sum over all strategies s of: 1 / (k + rank_s(d))
    where k is a constant (default 60) that prevents over-weighting top results.

    Example:
        >>> from ogbujipt.retrieval.hybrid import HybridSearch
        >>> from ogbujipt.retrieval.sparse import BM25Search
        >>> # Assume you have a dense vector search strategy too
        >>>
        >>> # Combine dense and sparse retrieval
        >>> hybrid = HybridSearch(strategies=[
        ...     dense_search,  # Your dense vector strategy
        ...     BM25Search()
        ... ])
        >>> results = hybrid.execute('machine learning', backends=[db], limit=10)
        >>> async for result in results:
        ...     print(f'{result.score:.3f}: {result.content[:50]}...')

    Note: RRF is particularly effective when combining complementary retrieval
    methods (e.g., semantic + keyword-based).
    '''

    def __init__(
        self,
        strategies: list[SearchStrategy],
        k: int = 60,  # RRF constant (typical range: 1-100)
        strategy_limits: dict[str, int] | None = None,  # Per-strategy result limits
    ):
        '''
        Initialize hybrid search with RRF.

        Args:
            strategies: List of search strategies to combine (e.g., [DenseSearch(), BM25Search()])
            k: RRF constant controlling rank weight distribution (default 60).
                Smaller k gives more weight to top-ranked results.
                Typical range: [1, 100]. Research suggests 60 works well across domains.
            strategy_limits: Optional dict mapping strategy names to result limits.
                E.g., {'BM25Search': 20, 'DenseSearch': 10} to fetch different amounts
                from each strategy before fusion. If not specified, uses the main limit
                for all strategies.
        '''
        if not strategies:
            raise ValueError('HybridSearch requires at least one strategy')

        self.strategies = strategies
        self.k = k
        self.strategy_limits = strategy_limits or {}

    async def execute(
        self,
        query: str,
        backends: list[KBBackend],
        limit: int = 5,
        threshold: float | None = None,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''
        Execute hybrid search using RRF to combine multiple strategies.

        Args:
            query: Search query string
            backends: List of KB backends to search
            limit: Maximum number of results to return after fusion
            threshold: Minimum RRF score threshold (optional)
            **kwargs: Additional options passed to individual strategies

        Yields:
            SearchResult objects sorted by RRF score (highest first)

        The RRF score is normalized to [0, 1] for consistency with other strategies.
        '''
        if not backends:
            logger.warning('hybrid_no_backends')
            return

        logger.info('hybrid_search_starting', strategy_count=len(self.strategies),
                   backend_count=len(backends))

        # Collect results from all strategies in parallel
        strategy_results = await asyncio.gather(*[
            self._run_strategy(strategy, query, backends, limit, **kwargs)
            for strategy in self.strategies
        ])

        # Build RRF scores by document content (using content as unique key)
        # Maps: content -> {score, metadata, source, contributing_strategies}
        rrf_scores = defaultdict(lambda: {
            'score': 0.0,
            'metadata': {},
            'source': set(),
            'ranks': []  # Track individual ranks for debugging
        })

        # Process each strategy's results
        for strategy_idx, results in enumerate(strategy_results):
            strategy_name = self.strategies[strategy_idx].__class__.__name__

            for rank, result in enumerate(results, start=1):
                # Calculate RRF contribution: 1 / (k + rank)
                rrf_contribution = 1.0 / (self.k + rank)

                doc_key = result.content  # Use content as key
                rrf_scores[doc_key]['score'] += rrf_contribution
                rrf_scores[doc_key]['ranks'].append((strategy_name, rank, result.score))

                # Merge metadata (later strategies override earlier ones if keys conflict)
                rrf_scores[doc_key]['metadata'].update(result.metadata)

                # Track all contributing sources
                rrf_scores[doc_key]['source'].add(result.source)

                # Keep original content and first-seen result for metadata
                if 'content' not in rrf_scores[doc_key]:
                    rrf_scores[doc_key]['content'] = result.content

            logger.debug('hybrid_strategy_processed', strategy=strategy_name,
                        result_count=len(results))

        # Sort by RRF score descending
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        if not sorted_docs:
            logger.info('hybrid_no_results')
            return

        # Normalize scores to [0, 1]
        max_rrf_score = sorted_docs[0][1]['score'] if sorted_docs else 1.0

        # Yield results up to limit
        returned = 0
        for content, doc_data in sorted_docs:
            if returned >= limit:
                break

            normalized_score = doc_data['score'] / max_rrf_score if max_rrf_score > 0 else 0.0

            # Apply threshold if specified
            if threshold is not None and normalized_score < threshold:
                continue

            # Combine sources into a readable string
            source_str = 'hybrid_rrf(' + ','.join(sorted(doc_data['source'])) + ')'

            # Add RRF metadata for transparency
            doc_data['metadata']['rrf_k'] = self.k
            doc_data['metadata']['rrf_raw_score'] = doc_data['score']
            doc_data['metadata']['rrf_ranks'] = doc_data['ranks']

            result = SearchResult(
                content=content,
                score=normalized_score,
                metadata=doc_data['metadata'],
                source=source_str
            )

            yield result
            returned += 1

        logger.info('hybrid_search_complete', total_candidates=len(sorted_docs),
                   returned=returned, max_rrf=max_rrf_score)

    async def _run_strategy(
        self,
        strategy: SearchStrategy,
        query: str,
        backends: list[KBBackend],
        limit: int,
        **kwargs
    ) -> list[SearchResult]:
        '''
        Run a single strategy and collect results.

        Args:
            strategy: Search strategy to execute
            query: Search query
            backends: KB backends
            limit: Result limit for this strategy
            **kwargs: Additional strategy options

        Returns:
            List of SearchResult objects
        '''
        strategy_name = strategy.__class__.__name__

        # Use per-strategy limit if configured
        strategy_limit = self.strategy_limits.get(strategy_name, limit * 2)  # Fetch more for better fusion

        try:
            results = []
            async for result in strategy.execute(
                query=query,
                backends=backends,
                limit=strategy_limit,
                **kwargs
            ):
                results.append(result)

            logger.debug('hybrid_strategy_executed', strategy=strategy_name,
                        result_count=len(results))
            return results

        except Exception as e:
            logger.error('hybrid_strategy_failed', strategy=strategy_name, error=str(e))
            return []  # Return empty list on failure, continue with other strategies


class RerankedHybridSearch:
    '''
    Hybrid search with reranking using cross-encoder models.

    First performs hybrid search (RRF fusion), then reranks the top-K candidates
    using a more powerful cross-encoder model for final ranking.

    Two-stage retrieval:
    1. Initial retrieval: Fast RRF fusion of multiple strategies (dense + sparse)
    2. Reranking: Slower but more accurate cross-encoder scoring of top candidates

    This pattern is more efficient than using cross-encoders for initial retrieval,
    since reranking only scores a small set of candidates.

    Example:
        >>> from ogbujipt.retrieval.hybrid import RerankedHybridSearch
        >>> from ogbujipt.retrieval.sparse import BM25Search
        >>> from rerankers import Reranker
        >>>
        >>> # Initialize reranker (e.g., HuggingFace cross-encoder)
        >>> reranker = Reranker(model_name='BAAI/bge-reranker-base')
        >>>
        >>> # Combine hybrid search with reranking
        >>> reranked = RerankedHybridSearch(
        ...     strategies=[dense_search, BM25Search()],
        ...     reranker=reranker,
        ...     rerank_top_k=20  # Rerank top 20 from initial retrieval
        ... )
        >>> results = reranked.execute('machine learning', backends=[db], limit=5)
        >>> async for result in results:
        ...     print(f'{result.score:.3f}: {result.content[:50]}...')

    Note: Install rerankers with: pip install "rerankers[transformers]"

    For some models, you may need special configuration:
        >>> # ZeRank-2 requires trust_remote_code and batch_size=1 (padding token issue)
        >>> reranker = Reranker(
        ...     model_name='zeroentropy/zerank-2',
        ...     model_kwargs={'trust_remote_code': True},
        ...     batch_size=1  # Required: model lacks padding token
        ... )

    If you get "Cannot handle batch sizes > 1 if no padding token is defined",
    set batch_size=1 when creating the Reranker.
    '''
    # To-do: Articulate support for the likes of https://huggingface.co/lightblue/lb-reranker-0.5B-v1.0
    def __init__(
        self,
        strategies: list[SearchStrategy],
        reranker,  # rerankers.Reranker instance
        rerank_top_k: int = 50,  # Number of candidates to rerank
        k: int = 60,  # RRF constant for initial fusion
        strategy_limits: dict[str, int] | None = None,
    ):
        '''
        Initialize reranked hybrid search.

        Args:
            strategies: List of search strategies to combine (e.g., [DenseSearch(), BM25Search()])
            reranker: A rerankers.Reranker instance (e.g., cross-encoder, Cohere API, etc.)
            rerank_top_k: Number of top candidates from initial retrieval to rerank.
                Should be larger than your final limit to give reranker enough candidates.
                Typical: 20-100 depending on corpus size and compute budget.
            k: RRF constant for initial fusion (default 60)
            strategy_limits: Optional per-strategy result limits for initial retrieval
        '''
        if not strategies:
            raise ValueError('RerankedHybridSearch requires at least one strategy')

        # Use HybridSearch for initial retrieval
        self.hybrid_search = HybridSearch(
            strategies=strategies,
            k=k,
            strategy_limits=strategy_limits
        )
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

    async def execute(
        self,
        query: str,
        backends: list[KBBackend],
        limit: int = 5,
        threshold: float | None = None,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        '''
        Execute hybrid search with reranking.

        Args:
            query: Search query string
            backends: List of KB backends to search
            limit: Maximum number of results to return after reranking
            threshold: Minimum reranker score threshold (optional)
            **kwargs: Additional options passed to initial retrieval strategies

        Yields:
            SearchResult objects sorted by reranker score (highest first)

        The process:
        1. Initial retrieval: Get top rerank_top_k candidates using RRF
        2. Reranking: Score candidates with cross-encoder
        3. Final ranking: Return top limit results by reranker score
        '''
        if not backends:
            logger.warning('reranked_hybrid_no_backends')
            return

        logger.info('reranked_hybrid_starting', strategy_count=len(self.hybrid_search.strategies),
                   backend_count=len(backends), rerank_top_k=self.rerank_top_k)

        # Stage 1: Initial retrieval with RRF fusion
        # Fetch more candidates than final limit to give reranker good options
        initial_limit = max(self.rerank_top_k, limit * 2)
        candidates = []

        async for result in self.hybrid_search.execute(
            query=query,
            backends=backends,
            limit=initial_limit,
            threshold=None,  # No threshold for initial retrieval
            **kwargs
        ):
            candidates.append(result)

        if not candidates:
            logger.info('reranked_hybrid_no_candidates')
            return

        logger.debug('reranked_hybrid_initial_retrieval', candidate_count=len(candidates))

        # Stage 2: Rerank using cross-encoder
        # Build documents for reranker (format expected by rerankers library)
        from rerankers import Document

        # Pass explicit doc_ids to preserve mapping to original candidates
        docs = [
            Document(text=result.content, doc_id=i, metadata=result.metadata)
            for i, result in enumerate(candidates)
        ]

        # Rerank with query
        try:
            reranked_results = self.reranker.rank(query=query, docs=docs)
            logger.debug('reranked_hybrid_reranking_complete', result_count=len(reranked_results.results))
        except Exception as e:
            logger.error('reranked_hybrid_reranking_failed', error=str(e))
            # Fallback to original RRF ordering if reranking fails
            logger.warning('reranked_hybrid_fallback_to_rrf')
            for i, result in enumerate(candidates[:limit]):
                if threshold is None or result.score >= threshold:
                    yield result
            return

        # Stage 3: Yield reranked results
        returned = 0
        for reranked_doc in reranked_results.results:
            if returned >= limit:
                break

            # Apply threshold if specified
            if threshold is not None and reranked_doc.score < threshold:
                continue

            # Find original result using doc_id we assigned
            original_idx = reranked_doc.doc_id
            if original_idx is None or original_idx >= len(candidates):
                logger.warning('reranked_hybrid_invalid_doc_id', doc_id=original_idx)
                continue

            original_result = candidates[original_idx]

            # Create new result with reranker score
            result = SearchResult(
                content=original_result.content,
                score=reranked_doc.score,  # Use reranker score
                metadata={
                    **original_result.metadata,
                    'reranker_score': reranked_doc.score,
                    'rrf_score': original_result.score,  # Keep original RRF score
                    'original_rank': original_idx + 1,
                },
                source=f'reranked({original_result.source})'
            )

            yield result
            returned += 1

        logger.info('reranked_hybrid_complete', total_candidates=len(candidates),
                   reranked=len(reranked_results.results), returned=returned)


__all__ = ['HybridSearch', 'RerankedHybridSearch']
