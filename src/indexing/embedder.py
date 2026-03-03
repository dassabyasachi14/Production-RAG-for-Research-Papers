"""
Google gemini-embedding-001 wrapper via the Generative Language REST API.

Uses task_type="RETRIEVAL_DOCUMENT" for indexing and "RETRIEVAL_QUERY"
for search queries (asymmetric retrieval).

Note: text-embedding-004 and embedding-001 were deprecated and removed by
Google. gemini-embedding-001 is the current recommended replacement.

Output dimensionality: 3072 (default).
Free tier: 100 RPM, 1,500 requests/day.
"""

from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-embedding-001"
_BATCH_SIZE = 100  # batchEmbedContents limit
_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class Embedder:
    """Calls Google gemini-embedding-001 directly via the REST API."""

    def __init__(self, api_key: str, model_name: str = _DEFAULT_MODEL) -> None:
        self._api_key = api_key
        self.model_name = model_name
        logger.info("Embedder initialised with model=%s.", model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{_API_BASE}/{self.model_name}:{endpoint}?key={self._api_key}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding API error {exc.code} {exc.reason}: {detail}"
            ) from exc

    def _batch_embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Call batchEmbedContents; returns list of float vectors."""
        payload = {
            "requests": [
                {
                    "model": f"models/{self.model_name}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                }
                for t in texts
            ]
        }
        data = self._post("batchEmbedContents", payload)
        return [emb["values"] for emb in data["embeddings"]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of document texts (RETRIEVAL_DOCUMENT task type).

        Splits into batches of 100 to respect the API limit.

        Returns:
            np.ndarray of shape (N, 3072), dtype float32.
        """
        if not texts:
            return np.empty((0, 3072), dtype=np.float32)

        all_embeddings: List[List[float]] = []
        for i in range(math.ceil(len(texts) / _BATCH_SIZE)):
            batch = texts[i * _BATCH_SIZE : (i + 1) * _BATCH_SIZE]
            all_embeddings.extend(self._batch_embed(batch, "RETRIEVAL_DOCUMENT"))
        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string (RETRIEVAL_QUERY task type).

        Returns:
            np.ndarray of shape (3072,), dtype float32.
        """
        payload = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": query}]},
            "taskType": "RETRIEVAL_QUERY",
        }
        data = self._post("embedContent", payload)
        return np.array(data["embedding"]["values"], dtype=np.float32)
