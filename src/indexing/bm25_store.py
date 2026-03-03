"""
BM25 lexical search index with pickle-based persistence.

Each document gets its own BM25Okapi index, stored as a pickle file at
{index_dir}/{doc_id}.pkl. Indexes are loaded on demand and kept in memory
for the lifetime of the object.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

from src.utils.models import DocumentChunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class BM25Store:
    """Manages per-document BM25 indexes for lexical search."""

    def __init__(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        self.index_dir = index_dir
        # {doc_id: (BM25Okapi_instance, List[DocumentChunk])}
        self._indexes: Dict[str, Tuple[object, List[DocumentChunk]]] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(self, doc_id: str, chunks: List[DocumentChunk]) -> None:
        """
        Build a BM25 index for `doc_id` from the given chunks and persist it.

        If an index already exists for this doc_id it is overwritten.

        Args:
            doc_id: Unique identifier for the document.
            chunks: List of DocumentChunk objects to index.
        """
        from rank_bm25 import BM25Okapi

        tokenized = [_tokenize(c.content) for c in chunks]
        index = BM25Okapi(tokenized)
        self._indexes[doc_id] = (index, chunks)

        path = self._index_path(doc_id)
        with open(path, "wb") as f:
            pickle.dump({"index": index, "chunks": chunks}, f)

        logger.info(
            "Built and saved BM25 index for doc_id=%s (%d chunks).", doc_id, len(chunks)
        )

    def load_index(self, doc_id: str) -> bool:
        """
        Load a previously built BM25 index from disk into memory.

        Args:
            doc_id: Document identifier to load.

        Returns:
            True if loaded successfully, False if no index file exists.
        """
        path = self._index_path(doc_id)
        if not os.path.exists(path):
            logger.warning("No BM25 index file for doc_id=%s.", doc_id)
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)
        self._indexes[doc_id] = (data["index"], data["chunks"])
        logger.info("Loaded BM25 index for doc_id=%s.", doc_id)
        return True

    def load_all_indexes(self) -> None:
        """Load all BM25 indexes found in the index directory."""
        for fname in os.listdir(self.index_dir):
            if fname.endswith(".pkl"):
                doc_id = fname[: -len(".pkl")]
                if doc_id not in self._indexes:
                    self.load_index(doc_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        doc_id: Optional[str] = None,
        n_results: int = 30,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        BM25 search over indexed chunks.

        Args:
            query: The search query string.
            doc_id: If provided, search only that document's index.
                    If None, search all loaded indexes.
            n_results: Maximum number of results to return.

        Returns:
            List of (DocumentChunk, bm25_score) tuples ordered by
            descending BM25 score, truncated to n_results.
        """
        tokenized_query = _tokenize(query)
        results: List[Tuple[DocumentChunk, float]] = []

        target_ids = [doc_id] if doc_id else list(self._indexes.keys())

        for did in target_ids:
            if did not in self._indexes:
                # Attempt lazy load
                if not self.load_index(did):
                    continue

            index, chunks = self._indexes[did]
            scores = index.get_scores(tokenized_query)

            for chunk, score in zip(chunks, scores):
                if score > 0:
                    results.append((chunk, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_results]

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> None:
        """Remove a document's BM25 index from memory and disk."""
        self._indexes.pop(doc_id, None)
        path = self._index_path(doc_id)
        if os.path.exists(path):
            os.remove(path)
            logger.info("Deleted BM25 index for doc_id=%s.", doc_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _index_path(self, doc_id: str) -> str:
        return os.path.join(self.index_dir, f"{doc_id}.pkl")

    def list_indexed_doc_ids(self) -> List[str]:
        """Return all doc_ids with a persisted BM25 index file."""
        return [
            f[: -len(".pkl")]
            for f in os.listdir(self.index_dir)
            if f.endswith(".pkl")
        ]
