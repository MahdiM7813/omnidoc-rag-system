"""Application service layer used by FastAPI."""

from __future__ import annotations

import logging
from functools import lru_cache

from core.config import Settings, load_settings
from pipelines.conversation import ConversationService
from pipelines.ingestion import IngestionPipeline
from pipelines.rag import RagPipeline

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once for the API process."""
    return load_settings()


class RagApplicationService:
    """Coordinates API calls with ingestion and query pipelines."""

    def __init__(self, settings: Settings):
        """Initialize application service."""
        self.settings = settings
        self._pipeline: RagPipeline | None = None

    def ingest(self, paths: list[str]) -> str:
        """Run ingestion and invalidate the query pipeline."""
        IngestionPipeline(self.settings).ingest(paths)
        self._pipeline = None
        return f"Ingested {len(paths)} path(s) and rebuilt the index."

    def query(self, question: str, session_id: str | None = None):
        """Answer a query."""
        return self.pipeline.ask(question, session_id=session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear a conversation session."""
        ConversationService(self.settings, self.pipeline).clear_session(session_id)

    @property
    def pipeline(self) -> RagPipeline:
        """Lazy-load the RAG pipeline after the vector index exists."""
        if self._pipeline is None:
            self._pipeline = RagPipeline.from_config(self.settings)
        return self._pipeline


@lru_cache(maxsize=1)
def get_app_service() -> RagApplicationService:
    """Return cached application service."""
    return RagApplicationService(get_settings())
