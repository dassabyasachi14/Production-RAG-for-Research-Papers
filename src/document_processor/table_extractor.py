"""
Table extraction from PDFs using pdfplumber.

Detects and extracts tables on each page, converts them to markdown pipe-table
format, and returns them as DocumentChunks for indexing.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from src.utils.models import DocumentChunk

logger = logging.getLogger(__name__)

# Minimum table dimensions to process
MIN_ROWS = 2
MIN_COLS = 2


def extract_tables(
    pdf_path: str,
    doc_id: str,
    filename: str,
) -> List[DocumentChunk]:
    """
    Extract all tables from a PDF and return them as DocumentChunks.

    Uses pdfplumber for table detection. Each table is converted to a
    markdown pipe-table string. Tables with fewer than MIN_ROWS rows or
    MIN_COLS columns are skipped.

    Args:
        pdf_path: Path to the PDF file.
        doc_id: UUID string for the parent document.
        filename: Display name of the source PDF.

    Returns:
        List of DocumentChunk objects with content_type='table'.
    """
    import pdfplumber

    chunks: List[DocumentChunk] = []
    table_counter = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                if not _is_valid_table(table):
                    continue

                markdown = _table_to_markdown(table)
                if not markdown.strip():
                    continue

                table_counter += 1
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    filename=filename,
                    content=f"[Table {table_counter}]\n{markdown}",
                    content_type="table",
                    page_number=page_number,
                    section=None,
                    chunk_index=-1,  # Will be reassigned by chunker.build_chunks
                    metadata={
                        "table_index": table_counter,
                        "row_count": len(table),
                        "col_count": len(table[0]) if table else 0,
                    },
                )
                chunks.append(chunk)

    logger.info("Extracted %d tables from %s", len(chunks), filename)
    return chunks


def _is_valid_table(table: List[List[Optional[str]]]) -> bool:
    """Return True if the table meets minimum row and column thresholds."""
    if len(table) < MIN_ROWS:
        return False
    if not table[0] or len(table[0]) < MIN_COLS:
        return False
    return True


def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
    """
    Convert a pdfplumber table (list of rows) to a markdown pipe table.

    The first row is treated as the header. None cells are replaced with
    an empty string. Multi-line cell content is collapsed to a single line.
    """
    if not table:
        return ""

    def clean_cell(cell: Optional[str]) -> str:
        if cell is None:
            return ""
        return " ".join(str(cell).split())

    rows = [[clean_cell(cell) for cell in row] for row in table]
    col_count = max(len(row) for row in rows)

    # Pad rows to uniform column count
    rows = [row + [""] * (col_count - len(row)) for row in rows]

    header = rows[0]
    separator = ["---"] * col_count
    body = rows[1:]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(separator) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
