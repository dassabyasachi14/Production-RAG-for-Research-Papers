"""
Recursive text chunking with token-accurate splitting.

Merges text chunks (from pdf_extractor), image description chunks
(from image_processor), and table chunks (from table_extractor) into
a single ordered list of DocumentChunks with proper chunk_index values.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Dict, List, Optional

from src.utils.models import DocumentChunk

logger = logging.getLogger(__name__)

# Section header patterns common in research papers
SECTION_PATTERNS = [
    re.compile(r"^#+\s*(abstract)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#+\s*(introduction)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#+\s*(related work)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#+\s*(background)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#+\s*(method(?:s|ology)?)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#+\s*(experiment(?:s|al setup)?)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#+\s*(result(?:s)?)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#+\s*(discussion)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#+\s*(conclusion(?:s)?)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#+\s*(reference(?:s)?)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#+\s*(appendix)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#+\s*(acknowledgement(?:s)?)", re.IGNORECASE | re.MULTILINE),
]

# Separators used for recursive splitting (in priority order)
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def infer_section(text: str) -> Optional[str]:
    """
    Heuristic detection of the research paper section a chunk belongs to.

    Checks the first 200 characters of the text against known section
    header patterns. Returns the matched section name or None.
    """
    head = text[:200]
    for pattern in SECTION_PATTERNS:
        match = pattern.search(head)
        if match:
            return match.group(1).capitalize()
    return None


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Return the number of tokens in text using the specified tiktoken encoding."""
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def recursive_chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    encoding_name: str = "cl100k_base",
) -> List[str]:
    """
    Split text into chunks of at most `chunk_size` tokens using recursive
    separator splitting. Adds `overlap` tokens of context between chunks.

    Splitting order: double-newline → newline → period-space → space → character.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum tokens per chunk.
        overlap: Token overlap between consecutive chunks.
        encoding_name: tiktoken encoding to use for counting.

    Returns:
        List of text strings, each within the token budget.
    """
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)

    def _encode(t: str) -> List[int]:
        return enc.encode(t)

    def _decode(tokens: List[int]) -> str:
        return enc.decode(tokens)

    def _split(text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]
        sep = separators[0]
        if sep == "":
            # Character-level split of last resort
            tokens = _encode(text)
            chunks = []
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunks.append(_decode(tokens[start:end]))
                start = end - overlap if end - overlap > start else end
            return chunks

        parts = text.split(sep)
        result: List[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if count_tokens(candidate, encoding_name) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if count_tokens(part, encoding_name) > chunk_size:
                    result.extend(_split(part, separators[1:]))
                    current = ""
                else:
                    current = part
        if current:
            result.append(current)
        return result

    raw_chunks = _split(text, SEPARATORS)

    # Apply overlap: each chunk (except the first) starts with the tail of the previous
    if overlap <= 0 or len(raw_chunks) <= 1:
        return [c for c in raw_chunks if c.strip()]

    overlapped: List[str] = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tokens = _encode(overlapped[-1])
        tail_tokens = prev_tokens[-overlap:] if len(prev_tokens) > overlap else prev_tokens
        curr_tokens = _encode(raw_chunks[i])
        merged_tokens = tail_tokens + curr_tokens
        if len(merged_tokens) > chunk_size:
            merged_tokens = merged_tokens[:chunk_size]
        overlapped.append(_decode(merged_tokens))

    return [c for c in overlapped if c.strip()]


def build_chunks(
    doc_id: str,
    filename: str,
    page_texts: Dict[int, str],
    image_chunks: List[DocumentChunk],
    table_chunks: List[DocumentChunk],
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[DocumentChunk]:
    """
    Build a unified, ordered list of DocumentChunks from all content types.

    Text chunks are produced by recursively splitting each page's markdown text.
    Image and table chunks are inserted in page-number order alongside text chunks.
    All chunks receive a sequential `chunk_index` starting from 0.

    Args:
        doc_id: UUID string for the parent document.
        filename: Display name of the source PDF.
        page_texts: Dict[page_number, markdown_text] from pdf_extractor.
        image_chunks: List of image-description chunks from image_processor.
        table_chunks: List of table chunks from table_extractor.
        chunk_size: Max tokens per text chunk.
        overlap: Token overlap between consecutive text chunks.

    Returns:
        List[DocumentChunk] ordered by page then type (text, image, table).
    """
    # Index non-text chunks by page for interleaving
    images_by_page: Dict[int, List[DocumentChunk]] = {}
    for c in image_chunks:
        images_by_page.setdefault(c.page_number, []).append(c)

    tables_by_page: Dict[int, List[DocumentChunk]] = {}
    for c in table_chunks:
        tables_by_page.setdefault(c.page_number, []).append(c)

    all_chunks: List[DocumentChunk] = []
    chunk_index = 0

    for page_number in sorted(page_texts.keys()):
        page_text = page_texts[page_number]

        # Text chunks
        text_parts = recursive_chunk_text(page_text, chunk_size, overlap)
        for text in text_parts:
            if not text.strip():
                continue
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                filename=filename,
                content=text,
                content_type="text",
                page_number=page_number,
                section=infer_section(text),
                chunk_index=chunk_index,
                metadata={},
            )
            all_chunks.append(chunk)
            chunk_index += 1

        # Image description chunks for this page
        for img_chunk in images_by_page.get(page_number, []):
            img_chunk = img_chunk.model_copy(update={"chunk_index": chunk_index})
            all_chunks.append(img_chunk)
            chunk_index += 1

        # Table chunks for this page
        for tbl_chunk in tables_by_page.get(page_number, []):
            tbl_chunk = tbl_chunk.model_copy(update={"chunk_index": chunk_index})
            all_chunks.append(tbl_chunk)
            chunk_index += 1

    logger.info(
        "Built %d chunks from %s (%d text, %d images, %d tables)",
        len(all_chunks),
        filename,
        sum(1 for c in all_chunks if c.content_type == "text"),
        sum(1 for c in all_chunks if c.content_type == "image_description"),
        sum(1 for c in all_chunks if c.content_type == "table"),
    )
    return all_chunks
