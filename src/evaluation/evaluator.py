"""
LLM-as-judge scoring for RAG evaluation.

Uses Gemini 2.5 Flash (via the existing LLMClient) to score each
(question, answer, context) triple on three independent metrics:

  Faithfulness     — fraction of answer claims supported by the retrieved context
  Answer Relevancy — how directly the answer addresses the question
  Context Precision — fraction of retrieved chunks that are genuinely relevant
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.generation.llm_client import LLMClient
    from src.generation.prompt_manager import PromptManager
    from src.utils.models import RetrievedChunk

logger = logging.getLogger(__name__)


class EvalSample(BaseModel):
    """One evaluated question/answer pair with all three metric scores."""

    question: str
    answer: str
    is_grounded: bool
    retrieved_chunks: List  # List[RetrievedChunk] — avoid circular import
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None


class Evaluator:
    """Scores EvalSample objects using Gemini as an LLM judge."""

    def __init__(
        self,
        llm_client: "LLMClient",
        prompt_manager: "PromptManager",
    ) -> None:
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager

    # ------------------------------------------------------------------
    # Individual metric scorers
    # ------------------------------------------------------------------

    def score_faithfulness(self, sample: EvalSample) -> float:
        """
        Score the fraction of factual claims in the answer that are
        supported by the retrieved context.

        Returns 0.0 for non-grounded answers (INSUFFICIENT_EVIDENCE).
        Returns float in [0.0, 1.0].
        """
        if not sample.is_grounded or not sample.retrieved_chunks:
            return 0.0

        numbered_context = self._format_chunks(sample.retrieved_chunks)
        prompt_config = self.prompt_manager.load_prompt("eval_faithfulness")
        user_message = self.prompt_manager.render_template(
            prompt_config["user_template"],
            question=sample.question,
            answer=sample.answer,
            numbered_context=numbered_context,
        )
        raw = self.llm_client.generate(
            system_prompt=prompt_config["system_prompt"],
            user_message=user_message,
            temperature=0.0,
            max_tokens=256,
        )
        return self._parse_score(raw, "score")

    def score_answer_relevancy(self, sample: EvalSample) -> float:
        """
        Score how directly and completely the answer addresses the question.

        Returns 0.0 for non-grounded answers.
        Returns float in [0.0, 1.0].
        """
        if not sample.is_grounded:
            return 0.0

        prompt_config = self.prompt_manager.load_prompt("eval_answer_relevancy")
        user_message = self.prompt_manager.render_template(
            prompt_config["user_template"],
            question=sample.question,
            answer=sample.answer,
        )
        raw = self.llm_client.generate(
            system_prompt=prompt_config["system_prompt"],
            user_message=user_message,
            temperature=0.0,
            max_tokens=256,
        )
        return self._parse_score(raw, "score")

    def score_context_precision(self, sample: EvalSample) -> float:
        """
        Score the fraction of retrieved chunks that are relevant to the question.

        Runs regardless of is_grounded (measures retrieval quality independently).
        Returns float in [0.0, 1.0].
        """
        if not sample.retrieved_chunks:
            return 0.0

        numbered_context = self._format_chunks(sample.retrieved_chunks)
        prompt_config = self.prompt_manager.load_prompt("eval_context_precision")
        user_message = self.prompt_manager.render_template(
            prompt_config["user_template"],
            question=sample.question,
            numbered_context=numbered_context,
        )
        raw = self.llm_client.generate(
            system_prompt=prompt_config["system_prompt"],
            user_message=user_message,
            temperature=0.0,
            max_tokens=256,
        )
        return self._parse_context_precision(raw, total=len(sample.retrieved_chunks))

    def score(self, sample: EvalSample) -> EvalSample:
        """
        Score a sample on all three metrics and return the updated sample.

        Each scorer is called independently so a failure in one does not
        block the others.
        """
        faithfulness = self._safe_score(self.score_faithfulness, sample, "faithfulness")
        answer_relevancy = self._safe_score(
            self.score_answer_relevancy, sample, "answer_relevancy"
        )
        context_precision = self._safe_score(
            self.score_context_precision, sample, "context_precision"
        )
        return sample.model_copy(
            update={
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_score(self, scorer, sample: EvalSample, name: str) -> float:
        try:
            return scorer(sample)
        except Exception as exc:
            logger.warning("Scorer %s failed: %s", name, exc)
            return 0.0

    @staticmethod
    def _format_chunks(chunks) -> str:
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            lines.append(f"[{i}] {chunk.content[:300]}")
            lines.append("---")
        return "\n".join(lines)

    @staticmethod
    def _parse_score(raw: str, key: str) -> float:
        """Parse a float score from a JSON response like {"score": 0.85}."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        try:
            data = json.loads(text)
            value = float(data[key])
            return max(0.0, min(1.0, value))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # Fallback: scan for first float in [0,1] in the text
            import re
            matches = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", raw)
            if matches:
                return max(0.0, min(1.0, float(matches[0])))
            return 0.0

    @staticmethod
    def _parse_context_precision(raw: str, total: int) -> float:
        """Parse {"relevant_indices": [...], "total": N} and compute precision."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        try:
            data = json.loads(text)
            relevant = data.get("relevant_indices", [])
            n_total = data.get("total", total)
            if n_total == 0:
                return 0.0
            return max(0.0, min(1.0, len(relevant) / n_total))
        except (json.JSONDecodeError, TypeError, ZeroDivisionError):
            return 0.0
