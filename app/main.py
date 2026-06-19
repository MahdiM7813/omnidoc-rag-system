"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from app.schemas import HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.services import RagApplicationService, get_app_service, get_settings
from core.logging import setup_logging

settings = get_settings()
setup_logging(settings.logging.level, json_logs=settings.logging.json_logs, verbose=settings.app.verbose)

app = FastAPI(
    title=settings.app.name,
    version="0.1.0",
    description="Conversational multimodal RAG API for complex PDFs.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health."""
    return HealthResponse(service=settings.app.name)


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    service: RagApplicationService = Depends(get_app_service),
) -> IngestResponse:
    """Build or rebuild the retrieval index."""
    try:
        message = service.ingest(request.paths)
        return IngestResponse(status="ok", message=message)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    service: RagApplicationService = Depends(get_app_service),
) -> QueryResponse:
    """Ask a grounded RAG question."""
    try:
        response = service.query(request.question, session_id=request.session_id)
        return QueryResponse(
            answer=response.answer,
            question=response.question,
            rewritten_question=response.rewritten_question,
            sources=response.sources,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/sessions/{session_id}")
def clear_session(
    session_id: str,
    service: RagApplicationService = Depends(get_app_service),
) -> dict[str, str]:
    """Clear persisted chat memory for a session."""
    service.clear_session(session_id)
    return {"status": "ok", "session_id": session_id}
