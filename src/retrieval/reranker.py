"""
Cross-encoder re-ranking for precise chunk selection.

A bi-encoder (used in the hybrid retrieval stage) embeds query and
documents independently. A cross-encoder processes them jointly, giving
substantially more accurate relevance scores at the cost of higher latency.

The cross-encoder is loaded once and kept in memory — never reloaded
per request.
"""

from __future__ import annotations

import logging
from typing import List

from src.utils.models import RetrievedChunk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Re-ranks a list of RetrievedChunk objects using a cross-encoder model.

    The cross-encoder scores each (query, chunk_content) pair and the
    chunks are returned ordered by that score, truncated to top_k.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder model: %s", model_name)
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        logger.info("Cross-encoder model loaded.")

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 8,
    ) -> List[RetrievedChunk]:
        """
        Score and re-rank chunks by cross-encoder relevance.

        Args:
            query: The user's question.
            chunks: Candidate chunks from hybrid retrieval.
            top_k: Number of top chunks to return after reranking.

        Returns:
            List of RetrievedChunk, length <= top_k, ordered by
            descending cross-encoder score. Each chunk's retrieval_score
            is replaced by the cross-encoder score for downstream use.
        """
        if not chunks:
            return []

        pairs = [[query, chunk.content] for chunk in chunks]
        scores = self.model.predict(pairs)

        scored = sorted(
            zip(chunks, scores), key=lambda x: float(x[1]), reverse=True
        )
        top = scored[:top_k]

        reranked: List[RetrievedChunk] = []
        for rank, (chunk, score) in enumerate(top, start=1):
            reranked.append(
                chunk.model_copy(
                    update={
                        "retrieval_score": float(score),
                        "retrieval_rank": rank,
                    }
                )
            )

        logger.info(
            "Reranker: %d → top %d (best score=%.4f)",
            len(chunks),
            len(reranked),
            reranked[0].retrieval_score if reranked else 0.0,
        )
        return reranked
