"""
Hybrid retrieval combining BM25 lexical search and vector semantic search.

Results from both search modalities are fused using Reciprocal Rank Fusion
(RRF), which avoids incompatible score scales by operating on ranks rather
than raw scores.

RRF formula: score(d) = Σ 1 / (k + rank_i(d))
where k=60 is the standard smoothing constant.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.utils.models import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)

RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[str, float]]],
    k: int = RRF_K,
) -> Dict[str, float]:
    """
    Merge multiple ranked result lists into a single RRF-scored dict.

    Args:
        result_lists: Each list contains (chunk_id, score) tuples already
                      ordered by descending relevance for one retrieval modality.
        k: RRF smoothing constant (default 60).

    Returns:
        Dict mapping chunk_id to its aggregated RRF score.
    """
    rrf_scores: Dict[str, float] = {}
    for ranked_list in result_lists:
        for rank, (chunk_id, _score) in enumerate(ranked_list, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return rrf_scores


class HybridRetriever:
    """
    Retrieves candidate chunks via BM25 + vector search and fuses with RRF.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedder: Embedder,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        doc_id: Optional[str] = None,
        n_vector: int = 30,
        n_bm25: int = 30,
        n_final: int = 20,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query using hybrid search.

        Pipeline:
        1. Embed query → vector search → top n_vector results
        2. BM25 search → top n_bm25 results
        3. RRF fusion → top n_final candidates

        Args:
            query: User question or search string.
            doc_id: Optional document UUID to restrict search scope.
            n_vector: Candidates to fetch from vector search.
            n_bm25: Candidates to fetch from BM25 search.
            n_final: Final number of fused candidates to return.

        Returns:
            List of RetrievedChunk ordered by descending RRF score.
        """
        # --- Vector search ---
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.query(
            query_embedding, n_results=n_vector, doc_id=doc_id
        )
        vector_list: List[Tuple[str, float]] = [
            (chunk.chunk_id, score) for chunk, score in vector_results
        ]

        # --- BM25 search ---
        bm25_results = self.bm25_store.query(
            query, doc_id=doc_id, n_results=n_bm25
        )
        bm25_list: List[Tuple[str, float]] = [
            (chunk.chunk_id, score) for chunk, score in bm25_results
        ]

        # --- RRF fusion ---
        rrf_scores = reciprocal_rank_fusion([vector_list, bm25_list])

        # Build a lookup of chunk objects from both result sets
        chunk_lookup: Dict[str, DocumentChunk] = {}
        for chunk, _ in vector_results:
            chunk_lookup[chunk.chunk_id] = chunk
        for chunk, _ in bm25_results:
            chunk_lookup[chunk.chunk_id] = chunk

        # Sort by RRF score and take top n_final
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        top_ids = sorted_ids[:n_final]

        retrieved: List[RetrievedChunk] = []
        for rank, chunk_id in enumerate(top_ids, start=1):
            if chunk_id not in chunk_lookup:
                continue
            base = chunk_lookup[chunk_id]
            retrieved.append(
                RetrievedChunk(
                    **base.model_dump(),
                    retrieval_score=rrf_scores[chunk_id],
                    retrieval_rank=rank,
                )
            )

        logger.info(
            "HybridRetriever: vector=%d, bm25=%d → fused=%d (top %d returned)",
            len(vector_list),
            len(bm25_list),
            len(rrf_scores),
            len(retrieved),
        )
        return retrieved
