"""
Numpy + pickle vector store — zero external dependencies, Python 3.14 safe.

Stores embeddings and chunk metadata in a single pickle file. Cosine
similarity is computed with numpy. Suitable for research paper scale
(typically < 10,000 chunks per session).

Replaces ChromaDB which is incompatible with Python 3.14 (pydantic v1 issue).
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.models import DocumentChunk

logger = logging.getLogger(__name__)

_STORE_FILE = "vectors.pkl"


class VectorStore:
    """In-process cosine-similarity vector store with pickle persistence."""

    def __init__(self, persist_dir: str) -> None:
        os.makedirs(persist_dir, exist_ok=True)
        self._store_path = os.path.join(persist_dir, _STORE_FILE)
        # chunk_id → embedding (float32, dim set by the embedding model)
        self._vectors: Dict[str, np.ndarray] = {}
        # chunk_id → DocumentChunk
        self._chunks: Dict[str, DocumentChunk] = {}
        self._load()
        logger.info(
            "VectorStore initialised. %d chunks loaded from disk.",
            len(self._chunks),
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add_chunks(
        self, chunks: List[DocumentChunk], embeddings: np.ndarray
    ) -> None:
        """
        Store chunks and their embeddings.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: np.ndarray of shape (N, 768).
        """
        for chunk, emb in zip(chunks, embeddings):
            self._vectors[chunk.chunk_id] = emb.astype(np.float32)
            self._chunks[chunk.chunk_id] = chunk
        self._save()
        logger.info("Stored %d chunks in vector store.", len(chunks))

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a given document."""
        to_delete = [
            cid for cid, c in self._chunks.items() if c.doc_id == doc_id
        ]
        for cid in to_delete:
            del self._vectors[cid]
            del self._chunks[cid]
        self._save()
        logger.info(
            "Deleted %d chunks for doc_id=%s.", len(to_delete), doc_id
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 30,
        doc_id: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Cosine similarity search.

        Args:
            query_embedding: 1-D array of shape (768,).
            n_results: Maximum number of results.
            doc_id: If provided, restrict search to this document only.

        Returns:
            List of (DocumentChunk, cosine_similarity) tuples,
            ordered by descending similarity.
        """
        if not self._chunks:
            return []

        # Filter candidates
        if doc_id:
            cids = [
                cid
                for cid, c in self._chunks.items()
                if c.doc_id == doc_id
            ]
        else:
            cids = list(self._chunks.keys())

        if not cids:
            return []

        # Stack embeddings into matrix (N, 768)
        vecs = np.stack([self._vectors[cid] for cid in cids])

        # Normalise for cosine similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        v_norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        vecs_norm = vecs / v_norms

        sims = vecs_norm @ q_norm  # (N,)

        top_n = min(n_results, len(cids))
        top_idx = np.argsort(sims)[::-1][:top_n]

        return [
            (self._chunks[cids[i]], float(sims[i])) for i in top_idx
        ]

    def list_documents(self) -> List[dict]:
        """Return distinct documents as [{doc_id, filename}]."""
        seen: Dict[str, str] = {}
        for chunk in self._chunks.values():
            if chunk.doc_id not in seen:
                seen[chunk.doc_id] = chunk.filename
        return [{"doc_id": k, "filename": v} for k, v in seen.items()]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        with open(self._store_path, "wb") as f:
            pickle.dump(
                {"vectors": self._vectors, "chunks": self._chunks}, f
            )

    def _load(self) -> None:
        if os.path.exists(self._store_path):
            with open(self._store_path, "rb") as f:
                data = pickle.load(f)
            self._vectors = data.get("vectors", {})
            self._chunks = data.get("chunks", {})
