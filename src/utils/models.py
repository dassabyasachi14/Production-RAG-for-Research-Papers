from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A single chunk of content extracted from a PDF document."""

    chunk_id: str
    doc_id: str
    filename: str
    content: str
    content_type: Literal["text", "image_description", "table"]
    page_number: int
    section: Optional[str] = None
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(DocumentChunk):
    """A DocumentChunk enriched with retrieval scoring information."""

    retrieval_score: float
    retrieval_rank: int


class Citation(BaseModel):
    """A single citation linking an answer claim to a source chunk."""

    citation_index: int
    chunk_id: str
    filename: str
    page_number: int
    excerpt: str


class GeneratedAnswer(BaseModel):
    """The final answer produced by the RAG pipeline."""

    answer: str
    citations: List[Citation] = Field(default_factory=list)
    is_grounded: bool = True
    decline_reason: Optional[str] = None
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
