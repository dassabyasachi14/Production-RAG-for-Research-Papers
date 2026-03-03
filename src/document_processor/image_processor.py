"""
Image extraction from PDFs and description via Claude Vision.

Extracts embedded images using PyMuPDF, filters out small/decorative ones,
and calls the Claude claude-sonnet-4-6 vision API to generate textual descriptions
suitable for indexing in the RAG pipeline.
"""

from __future__ import annotations

import base64
import logging
import uuid
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from src.generation.llm_client import LLMClient
    from src.generation.prompt_manager import PromptManager

from src.utils.models import DocumentChunk

logger = logging.getLogger(__name__)

# Minimum image dimensions to process (skip decorative/icon images)
MIN_WIDTH = 50
MIN_HEIGHT = 50


def extract_and_describe_images(
    pdf_path: str,
    doc_id: str,
    filename: str,
    llm_client: "LLMClient",
    prompt_manager: "PromptManager",
    min_size: Tuple[int, int] = (MIN_WIDTH, MIN_HEIGHT),
) -> List[DocumentChunk]:
    """
    Extract images from a PDF and generate descriptions via Claude Vision.

    Each image that passes the size filter is sent to Claude claude-sonnet-4-6 with
    the image_description prompt and the resulting description is stored as a
    DocumentChunk with content_type='image_description'.

    Args:
        pdf_path: Path to the PDF file.
        doc_id: UUID string identifying the parent document.
        filename: Display name of the PDF file.
        llm_client: Initialised LLMClient for Claude API calls.
        prompt_manager: PromptManager for loading the image description prompt.
        min_size: (width, height) threshold — images smaller than this are skipped.

    Returns:
        List of DocumentChunk objects with image descriptions.
    """
    import fitz  # PyMuPDF

    prompt_config = prompt_manager.load_prompt("image_description")
    system_prompt = prompt_config["system_prompt"]

    doc = fitz.open(pdf_path)
    chunks: List[DocumentChunk] = []
    image_counter = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                image_bytes, media_type = _extract_image_bytes(doc, xref)
            except Exception as exc:
                logger.debug("Could not extract image xref=%d: %s", xref, exc)
                continue

            width, height = _get_image_dimensions(doc, xref)
            if width < min_size[0] or height < min_size[1]:
                logger.debug(
                    "Skipping small image (%dx%d) on page %d", width, height, page_number
                )
                continue

            try:
                description = llm_client.describe_image(
                    image_bytes=image_bytes,
                    system_prompt=system_prompt,
                    media_type=media_type,
                )
            except Exception as exc:
                logger.warning(
                    "Claude Vision failed for image on page %d: %s", page_number, exc
                )
                continue

            if not description.strip():
                continue

            image_counter += 1
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                filename=filename,
                content=f"[Figure {image_counter}] {description}",
                content_type="image_description",
                page_number=page_number,
                section=None,
                chunk_index=-1,  # Will be reassigned by chunker.build_chunks
                metadata={
                    "image_index": image_counter,
                    "image_width": width,
                    "image_height": height,
                },
            )
            chunks.append(chunk)

    doc.close()
    logger.info(
        "Extracted %d image descriptions from %s", len(chunks), filename
    )
    return chunks


def _extract_image_bytes(doc, xref: int) -> Tuple[bytes, str]:
    """Extract raw image bytes and media type from a PyMuPDF xref."""
    import fitz

    pix = fitz.Pixmap(doc, xref)
    if pix.n > 4:  # CMYK or similar — convert to RGB
        pix = fitz.Pixmap(fitz.csRGB, pix)
    image_bytes = pix.tobytes("png")
    return image_bytes, "image/png"


def _get_image_dimensions(doc, xref: int) -> Tuple[int, int]:
    """Return (width, height) of an image by its xref."""
    import fitz

    pix = fitz.Pixmap(doc, xref)
    return pix.width, pix.height
