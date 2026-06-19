"""RAG query pipeline."""

from __future__ import annotations

import logging
from typing import Any

from core.config import Settings
from core.models import RagResponse, Source
from modules.document_processing import format_context
from modules.generator import AnswerGenerator, HistoryAwareQueryRewriter
from modules.memory import LongTermMemoryStore, ShortTermMemory
from modules.vectorstores import VectorStoreManager

logger = logging.getLogger(__name__)


class RagPipeline:
    """Retriever/generator-separated RAG pipeline."""

    def __init__(
        self,
        settings: Settings,
        retriever: Any,
        generator: Any,
        query_rewriter: Any | None = None,
        memory_store: LongTermMemoryStore | None = None,
    ):
        """Initialize pipeline dependencies."""
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.query_rewriter = query_rewriter
        self.memory_store = memory_store

    @classmethod
    def from_config(cls, settings: Settings) -> "RagPipeline":
        """Build a production pipeline from persisted index and configured models."""
        manager = VectorStoreManager(settings)
        retriever = manager.as_retriever()
        generator = AnswerGenerator(settings)
        query_rewriter = HistoryAwareQueryRewriter(settings) if settings.conversation.enabled else None
        memory_store = LongTermMemoryStore(settings.conversation.long_term_memory_path)
        return cls(settings, retriever, generator, query_rewriter, memory_store)

    def ask(self, question: str, session_id: str | None = None) -> RagResponse:
        """Answer a user question using retrieval and grounded generation."""
        memory = self._load_memory(session_id)
        chat_history = memory.to_langchain_messages() if self.settings.conversation.enabled else []
        rewritten = self._rewrite(question, chat_history)
        retrieved_docs = self._retrieve(rewritten)
        context = format_context(
            retrieved_docs,
            include_tables=self.settings.rag.include_tables,
            include_images_as_text=self.settings.rag.include_images_as_text,
            max_original_chars=self.settings.document_processing.max_original_chars,
        )
        answer = self.generator.generate(question, context, chat_history=chat_history)
        memory.add(question, answer)
        self._save_memory(session_id, memory)
        return RagResponse(
            answer=answer,
            question=question,
            rewritten_question=rewritten if rewritten != question else None,
            sources=self._sources_from_documents(retrieved_docs),
        )

    def _rewrite(self, question: str, chat_history: list[Any]) -> str:
        if not self.query_rewriter or not chat_history:
            return question
        return str(self.query_rewriter.rewrite(question, chat_history))

    def _retrieve(self, question: str) -> list[Any]:
        if hasattr(self.retriever, "invoke"):
            return list(self.retriever.invoke(question))
        if hasattr(self.retriever, "get_relevant_documents"):
            return list(self.retriever.get_relevant_documents(question))
        raise TypeError("Retriever must expose invoke() or get_relevant_documents().")

    def _load_memory(self, session_id: str | None) -> ShortTermMemory:
        if not self.settings.conversation.enabled:
            return ShortTermMemory(max_turns=0)
        if session_id and self.memory_store:
            return self.memory_store.load_session(session_id, max_turns=self.settings.conversation.max_turns)
        return ShortTermMemory(max_turns=self.settings.conversation.max_turns)

    def _save_memory(self, session_id: str | None, memory: ShortTermMemory) -> None:
        if session_id and self.memory_store and self.settings.conversation.enabled:
            self.memory_store.save_session(session_id, memory)

    def _sources_from_documents(self, documents: list[Any]) -> list[Source]:
        if not self.settings.rag.return_sources:
            return []
        sources: list[Source] = []
        for doc in documents:
            metadata = getattr(doc, "metadata", {}) or {}
            sources.append(
                Source(
                    id=metadata.get("id"),
                    modality=metadata.get("modality", "text"),
                    source_file=metadata.get("source_file"),
                    page=metadata.get("page"),
                    summary=str(getattr(doc, "page_content", ""))[:500],
                    metadata={key: value for key, value in metadata.items() if key != "image_b64"},
                )
            )
        return sources
