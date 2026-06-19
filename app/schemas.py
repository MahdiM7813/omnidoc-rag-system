"""FastAPI request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.models import Source


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str = "ok"
    service: str


class IngestRequest(BaseModel):
    """Request to ingest one or more PDF paths visible to the server."""

    paths: list[str] = Field(..., description="PDF files or directories on the server.")


class IngestResponse(BaseModel):
    """Ingest response."""

    status: str
    message: str


class QueryRequest(BaseModel):
    """RAG query request."""

    question: str
    session_id: str | None = "default"


class QueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    question: str
    rewritten_question: str | None = None
    sources: list[Source] = Field(default_factory=list)
