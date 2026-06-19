"""Shared domain models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentModality(str, Enum):
    """Supported source document modalities."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class Source(BaseModel):
    """A source returned with an answer."""

    id: str | None = None
    modality: DocumentModality | str = DocumentModality.TEXT
    source_file: str | None = None
    page: int | None = None
    score: float | None = None
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagResponse(BaseModel):
    """RAG answer payload."""

    answer: str
    question: str
    rewritten_question: str | None = None
    sources: list[Source] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatTurn(BaseModel):
    """A single user/assistant exchange."""

    user: str
    assistant: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvaluationQuestion(BaseModel):
    """Question used for lightweight RAG evaluation."""

    id: str
    question: str
    expected_keywords: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Single evaluation run result."""

    id: str
    question: str
    answer: str
    contains_expected_keywords: bool
    source_count: int
    latency_seconds: float
    metadata: dict[str, Any] = Field(default_factory=dict)
