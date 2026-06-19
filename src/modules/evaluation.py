"""Lightweight evaluation helpers for RAG outputs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

import yaml

from core.models import EvaluationQuestion, EvaluationResult, RagResponse


class RagLike(Protocol):
    """Protocol for RAG pipelines used by the evaluator."""

    def ask(self, question: str, session_id: str | None = None) -> RagResponse:
        """Ask a question and return a RAG response."""


def load_evaluation_questions(path: str | Path) -> list[EvaluationQuestion]:
    """Load evaluation questions from YAML."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [EvaluationQuestion.model_validate(item) for item in payload.get("questions", [])]


def evaluate_answer(question: EvaluationQuestion, response: RagResponse, latency: float) -> EvaluationResult:
    """Score one answer with deterministic smoke-test heuristics."""
    answer_lower = response.answer.lower()
    contains_keywords = all(keyword.lower() in answer_lower for keyword in question.expected_keywords)
    return EvaluationResult(
        id=question.id,
        question=question.question,
        answer=response.answer,
        contains_expected_keywords=contains_keywords,
        source_count=len(response.sources),
        latency_seconds=latency,
        metadata={"rewritten_question": response.rewritten_question},
    )


def run_evaluation(
    pipeline: RagLike,
    questions: list[EvaluationQuestion],
    output_path: str | Path,
) -> list[EvaluationResult]:
    """Run an evaluation set and write JSONL results."""
    results: list[EvaluationResult] = []
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for question in questions:
            started = time.perf_counter()
            response = pipeline.ask(question.question, session_id="evaluation")
            result = evaluate_answer(question, response, time.perf_counter() - started)
            results.append(result)
            file.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return results
