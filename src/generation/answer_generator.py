"""
Citation-enforced answer generation.

Orchestrates the final step of the RAG pipeline:
1. Formats retrieved chunks as a numbered context block.
2. Loads the versioned RAG answer prompt.
3. Calls Claude and checks for the INSUFFICIENT_EVIDENCE marker.
4. Parses inline [N] citation markers and maps them to source chunks.
5. Returns a fully structured GeneratedAnswer.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from src.generation.llm_client import LLMClient
from src.generation.prompt_manager import PromptManager
from src.utils.models import Citation, GeneratedAnswer, RetrievedChunk

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
DECLINE_MARKER = "INSUFFICIENT_EVIDENCE"

# Minimum cross-encoder score to attempt answer generation.
# Below this threshold we decline without calling the LLM.
DEFAULT_LOW_SCORE_THRESHOLD = 0.0  # cross-encoder scores can be negative; let LLM decide


def format_numbered_context(chunks: List[RetrievedChunk]) -> str:
    """
    Render retrieved chunks as a numbered list for the prompt.

    Format:
        [1] (filename, p.N, Section):
        <content>
        ---
        [2] ...

    Args:
        chunks: Ordered list of RetrievedChunk objects (1-indexed in output).

    Returns:
        Multi-line string ready for insertion into the user prompt template.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        section_part = f", {chunk.section}" if chunk.section else ""
        header = f"[{i}] ({chunk.filename}, p.{chunk.page_number}{section_part})"
        lines.append(header)
        lines.append(chunk.content)
        lines.append("---")
    return "\n".join(lines)


def parse_citations(
    answer_text: str,
    chunks: List[RetrievedChunk],
) -> List[Citation]:
    """
    Extract [N] citation markers from the answer and map to source chunks.

    Only citations that reference a valid chunk index (1 to len(chunks))
    are included. Duplicate citation indices produce a single Citation entry.

    Args:
        answer_text: The LLM's raw response text.
        chunks: The ordered list of RetrievedChunks passed to the LLM (1-indexed).

    Returns:
        List of Citation objects, one per unique cited chunk.
    """
    cited_indices = set()
    for match in CITATION_PATTERN.finditer(answer_text):
        idx = int(match.group(1))
        if 1 <= idx <= len(chunks):
            cited_indices.add(idx)

    citations: List[Citation] = []
    for idx in sorted(cited_indices):
        chunk = chunks[idx - 1]
        excerpt = chunk.content[:200].rstrip()
        if len(chunk.content) > 200:
            excerpt += "…"
        citations.append(
            Citation(
                citation_index=idx,
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                excerpt=excerpt,
            )
        )
    return citations


class AnswerGenerator:
    """
    Generates citation-enforced answers using Claude via the RAG pipeline.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_manager: PromptManager,
        low_score_threshold: float = DEFAULT_LOW_SCORE_THRESHOLD,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.low_score_threshold = low_score_threshold

    def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> GeneratedAnswer:
        """
        Generate a grounded answer with citations for the given query.

        Decline logic (sets is_grounded=False without calling LLM):
        - No chunks retrieved.
        - Top chunk score is below low_score_threshold.

        LLM-level decline:
        - If Claude's response starts with INSUFFICIENT_EVIDENCE.

        Args:
            query: The user's question.
            chunks: Reranked RetrievedChunk list (best first).

        Returns:
            GeneratedAnswer with answer text, citations, and grounding status.
        """
        # --- Pre-flight decline checks ---
        if not chunks:
            return GeneratedAnswer(
                answer="I could not find any relevant content in the uploaded document to answer your question.",
                citations=[],
                is_grounded=False,
                decline_reason="No chunks were retrieved from the document.",
                retrieved_chunks=[],
            )

        if chunks[0].retrieval_score < self.low_score_threshold:
            return GeneratedAnswer(
                answer="The document does not appear to contain information relevant to your question.",
                citations=[],
                is_grounded=False,
                decline_reason=(
                    f"Top retrieval score ({chunks[0].retrieval_score:.4f}) "
                    f"is below the minimum threshold ({self.low_score_threshold})."
                ),
                retrieved_chunks=chunks,
            )

        # --- Build prompt ---
        prompt_config = self.prompt_manager.load_prompt("rag_answer")
        system_prompt: str = prompt_config["system_prompt"]
        user_template: str = prompt_config["user_template"]
        temperature: float = float(prompt_config.get("temperature", 0.0))
        max_tokens: int = int(prompt_config.get("max_tokens", 2048))

        numbered_context = format_numbered_context(chunks)
        user_message = self.prompt_manager.render_template(
            user_template,
            question=query,
            numbered_context=numbered_context,
        )

        # --- Call LLM ---
        raw_answer = self.llm_client.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # --- Handle LLM-level decline ---
        if raw_answer.strip().startswith(DECLINE_MARKER):
            reason_line = raw_answer.strip().split("\n")[0]
            reason = reason_line.replace(f"{DECLINE_MARKER}:", "").strip()
            return GeneratedAnswer(
                answer=(
                    "I was unable to find sufficient evidence in the document "
                    "to answer your question reliably.\n\n"
                    f"**Reason:** {reason}"
                ),
                citations=[],
                is_grounded=False,
                decline_reason=reason,
                retrieved_chunks=chunks,
            )

        # --- Parse citations ---
        citations = parse_citations(raw_answer, chunks)

        return GeneratedAnswer(
            answer=raw_answer,
            citations=citations,
            is_grounded=True,
            decline_reason=None,
            retrieved_chunks=chunks,
        )
