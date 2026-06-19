"""Session-oriented conversational pipeline helpers."""

from __future__ import annotations

from core.config import Settings
from modules.memory import LongTermMemoryStore
from pipelines.rag import RagPipeline


class ConversationService:
    """Small service wrapper around RAG sessions."""

    def __init__(self, settings: Settings, pipeline: RagPipeline):
        """Initialize service."""
        self.settings = settings
        self.pipeline = pipeline
        self.memory_store = LongTermMemoryStore(settings.conversation.long_term_memory_path)

    def clear_session(self, session_id: str) -> None:
        """Clear persisted memory for a session."""
        self.memory_store.clear_session(session_id)
