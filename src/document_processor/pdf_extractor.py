"""
PDF text extraction using pymupdf4llm (primary) with PyMuPDF fallback.
Returns per-page markdown-formatted text for downstream chunking.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def extract_text_by_page(pdf_path: str) -> Dict[int, str]:
    """
    Extract markdown-formatted text from a PDF, keyed by page number (1-indexed).

    Uses pymupdf4llm for rich markdown output. Falls back to raw PyMuPDF
    plain-text extraction if pymupdf4llm fails.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Dict mapping page_number (int, 1-indexed) to markdown text (str).
    """
    try:
        return _extract_with_pymupdf4llm(pdf_path)
    except Exception as exc:
        logger.warning(
            "pymupdf4llm extraction failed (%s). Falling back to PyMuPDF.", exc
        )
        return _extract_with_pymupdf_fallback(pdf_path)


def _extract_with_pymupdf4llm(pdf_path: str) -> Dict[int, str]:
    import pymupdf4llm

    page_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    result: Dict[int, str] = {}
    for chunk in page_chunks:
        page_num = chunk.get("metadata", {}).get("page", 0) + 1
        text = chunk.get("text", "")
        if text.strip():
            result[page_num] = text
    return result


def _extract_with_pymupdf_fallback(pdf_path: str) -> Dict[int, str]:
    import fitz  # PyMuPDF

    result: Dict[int, str] = {}
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        if text.strip():
            result[page_index + 1] = text
    doc.close()
    return result


def extract_metadata(pdf_path: str) -> Dict[str, Optional[str]]:
    """
    Extract document-level metadata (title, author, subject) from the PDF.

    Returns:
        Dict with keys: title, author, subject, creation_date.
        Values are None if not available.
    """
    import fitz

    doc = fitz.open(pdf_path)
    raw = doc.metadata or {}
    doc.close()
    return {
        "title": raw.get("title") or None,
        "author": raw.get("author") or None,
        "subject": raw.get("subject") or None,
        "creation_date": raw.get("creationDate") or None,
    }
