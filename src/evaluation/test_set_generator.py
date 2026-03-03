"""
Test question generation for RAG evaluation.

Samples text chunks from the indexed document and uses Gemini to generate
factual, answerable questions suitable for evaluating the RAG pipeline.
"""

from __future__ import annotations

import json
import logging
import random
from typing import TYPE_CHECKING, List

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.generation.llm_client import LLMClient
    from src.generation.prompt_manager import PromptManager
    from src.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)


class TestCase(BaseModel):
    """A single evaluation question with its source chunk reference."""

    question: str
    source_chunk_id: str
    source_content: str  # First 300 chars of the chunk it was generated from


class TestSetGenerator:
    """Generates evaluation test questions from indexed document chunks."""

    def __init__(
        self,
        llm_client: "LLMClient",
        prompt_manager: "PromptManager",
        vector_store: "VectorStore",
    ) -> None:
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.vector_store = vector_store

    def generate(self, doc_id: str, n_questions: int = 10) -> List[TestCase]:
        """
        Generate n_questions factual evaluation questions from a document.

        Samples text chunks from the vector store for the given doc_id,
        passes them to Gemini with the eval_question_gen prompt, and parses
        the resulting JSON question list.

        Args:
            doc_id: UUID of the document to generate questions for.
            n_questions: Number of questions to generate.

        Returns:
            List of TestCase objects.
        """
        # Collect text chunks for this document
        text_chunks = [
            chunk
            for chunk in self.vector_store._chunks.values()
            if chunk.doc_id == doc_id and chunk.content_type == "text"
        ]

        if not text_chunks:
            logger.warning("No text chunks found for doc_id=%s", doc_id)
            return []

        # Sample a diverse set of chunks to cover the document
        sample_size = min(n_questions * 2, len(text_chunks))
        sampled = random.sample(text_chunks, sample_size)

        # Build context block from sampled chunks
        context_parts = []
        for i, chunk in enumerate(sampled, start=1):
            section = f" [{chunk.section}]" if chunk.section else ""
            context_parts.append(
                f"[Excerpt {i} — p.{chunk.page_number}{section}]\n{chunk.content[:400]}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Call LLM to generate questions
        prompt_config = self.prompt_manager.load_prompt("eval_question_gen")
        system_prompt: str = prompt_config["system_prompt"]
        user_template: str = prompt_config["user_template"]

        user_message = self.prompt_manager.render_template(
            user_template,
            n_questions=n_questions,
            context=context,
        )

        raw = self.llm_client.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.7,
            max_tokens=1024,
        )

        questions = self._parse_questions(raw, n_questions)

        # Map each question back to the most relevant sampled chunk (round-robin)
        test_cases: List[TestCase] = []
        for i, question in enumerate(questions):
            source_chunk = sampled[i % len(sampled)]
            test_cases.append(
                TestCase(
                    question=question,
                    source_chunk_id=source_chunk.chunk_id,
                    source_content=source_chunk.content[:300],
                )
            )

        logger.info(
            "Generated %d test questions for doc_id=%s", len(test_cases), doc_id
        )
        return test_cases

    def _parse_questions(self, raw: str, n_questions: int) -> List[str]:
        """
        Extract a list of question strings from the LLM's JSON response.

        Tries three strategies in order:
        1. Parse the entire response as a JSON array (clean response).
        2. Find the first JSON array anywhere in the response using regex
           (handles responses with surrounding prose).
        3. Line-by-line fallback: extract lines that look like questions.
        """
        import re

        # --- Strategy 1: whole response is a JSON array (possibly code-fenced) ---
        text = raw.strip()
        if text.startswith("```"):
            inner = text.splitlines()
            # Drop opening fence (and optional language tag) + closing fence
            start = 1
            end = len(inner) - 1 if inner[-1].strip() == "```" else len(inner)
            text = "\n".join(inner[start:end])

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                questions = [str(q).strip() for q in parsed if str(q).strip()]
                if questions:
                    return questions[:n_questions]
        except (json.JSONDecodeError, ValueError):
            pass

        # --- Strategy 2: find first JSON array anywhere in the raw response ---
        # Regex finds the first [...] block (non-greedy, handles multi-line)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    questions = [str(q).strip() for q in parsed if str(q).strip()]
                    if questions:
                        return questions[:n_questions]
            except (json.JSONDecodeError, ValueError):
                pass

        # --- Strategy 3: line-by-line extraction ---
        questions = []
        for line in raw.splitlines():
            line = line.strip()
            # Strip leading numbering like "1.", "1)", "-", "*"
            line = re.sub(r"^[\d]+[.)]\s*", "", line)
            line = line.lstrip("-*• ").strip().strip('"').strip("'").strip()
            if line.endswith("?") and len(line) > 15:
                questions.append(line)
        if questions:
            return questions[:n_questions]

        logger.warning(
            "Could not parse questions from LLM response. Raw response (first 300 chars): %s",
            raw[:300],
        )
        return []
