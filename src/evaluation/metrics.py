"""
RAG evaluation orchestration and reporting.

Runs the full pipeline (retrieve → rerank → generate → judge) for each
test question and aggregates the results into a structured EvalReport.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from pydantic import BaseModel

from src.evaluation.evaluator import EvalSample, Evaluator
from src.evaluation.test_set_generator import TestCase

if TYPE_CHECKING:
    from src.generation.answer_generator import AnswerGenerator
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class EvalReport(BaseModel):
    """Aggregated evaluation results for a single document."""

    doc_id: str
    filename: str
    n_questions: int
    n_grounded: int           # Answers where is_grounded=True
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    samples: List[EvalSample]


def run_evaluation(
    test_cases: List[TestCase],
    retriever: "HybridRetriever",
    reranker: "CrossEncoderReranker",
    generator: "AnswerGenerator",
    evaluator: Evaluator,
    doc_id: str,
    filename: str,
) -> EvalReport:
    """
    Run the full RAG pipeline and LLM-judge scoring for every test question.

    For each TestCase:
      1. Retrieve candidates with HybridRetriever
      2. Rerank with CrossEncoderReranker
      3. Generate answer with AnswerGenerator
      4. Score with Evaluator (faithfulness, answer_relevancy, context_precision)

    Args:
        test_cases: List of generated test questions.
        retriever:  Initialised HybridRetriever.
        reranker:   Initialised CrossEncoderReranker.
        generator:  Initialised AnswerGenerator.
        evaluator:  Initialised Evaluator.
        doc_id:     Document UUID to scope retrieval.
        filename:   Display name of the document.

    Returns:
        EvalReport with per-sample scores and aggregate means.
    """
    samples: List[EvalSample] = []

    for i, test_case in enumerate(test_cases, start=1):
        question = test_case.question
        logger.info(
            "Evaluating question %d/%d: %s", i, len(test_cases), question[:80]
        )

        try:
            # --- Retrieval ---
            retrieved = retriever.retrieve(query=question, doc_id=doc_id)
            reranked = reranker.rerank(query=question, chunks=retrieved)

            # --- Generation ---
            gen_answer = generator.generate(query=question, chunks=reranked)

            # --- Build sample ---
            sample = EvalSample(
                question=question,
                answer=gen_answer.answer,
                is_grounded=gen_answer.is_grounded,
                retrieved_chunks=reranked,
            )

            # --- Score ---
            scored = evaluator.score(sample)
            samples.append(scored)

        except Exception as exc:
            logger.warning("Failed to evaluate question %d: %s", i, exc)
            # Add a zeroed-out sample so the question still appears in the report
            samples.append(
                EvalSample(
                    question=question,
                    answer=f"[Error: {exc}]",
                    is_grounded=False,
                    retrieved_chunks=[],
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_precision=0.0,
                )
            )

    # --- Aggregate ---
    n = len(samples)
    n_grounded = sum(1 for s in samples if s.is_grounded)

    def _mean(attr: str) -> float:
        values = [getattr(s, attr) for s in samples if getattr(s, attr) is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    report = EvalReport(
        doc_id=doc_id,
        filename=filename,
        n_questions=n,
        n_grounded=n_grounded,
        mean_faithfulness=_mean("faithfulness"),
        mean_answer_relevancy=_mean("answer_relevancy"),
        mean_context_precision=_mean("context_precision"),
        samples=samples,
    )

    logger.info(
        "Evaluation complete. n=%d, grounded=%d, "
        "faithfulness=%.2f, relevancy=%.2f, precision=%.2f",
        n,
        n_grounded,
        report.mean_faithfulness,
        report.mean_answer_relevancy,
        report.mean_context_precision,
    )
    return report
