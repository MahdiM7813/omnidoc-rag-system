"""Vector store abstraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.config import Settings
from core.exceptions import ConfigurationError, DependencyNotInstalledError
from modules.embeddings import create_embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Build, persist, load, and expose retrievers for vector stores."""

    def __init__(self, settings: Settings):
        """Initialize the manager."""
        self.settings = settings
        self.embeddings = create_embeddings(settings.embeddings)

    def build(self, documents: list[Any]) -> Any:
        """Build and persist a vector store from LangChain documents."""
        provider = self.settings.vectorstore.provider
        if provider == "chroma":
            return self._build_chroma(documents)
        if provider == "faiss":
            return self._build_faiss(documents)
        raise ConfigurationError(f"Unsupported vectorstore provider: {provider}")

    def load(self) -> Any:
        """Load a persisted vector store."""
        provider = self.settings.vectorstore.provider
        if provider == "chroma":
            return self._load_chroma()
        if provider == "faiss":
            return self._load_faiss()
        raise ConfigurationError(f"Unsupported vectorstore provider: {provider}")

    def as_retriever(self, vectorstore: Any | None = None) -> Any:
        """Return a retriever from a vector store."""
        store = vectorstore or self.load()
        search_kwargs = {"k": self.settings.retrieval.top_k}
        if self.settings.retrieval.score_threshold is not None:
            search_kwargs["score_threshold"] = self.settings.retrieval.score_threshold
        return store.as_retriever(search_kwargs=search_kwargs)

    def _build_chroma(self, documents: list[Any]) -> Any:
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError as exc:
            raise DependencyNotInstalledError("Install chromadb and langchain-community.") from exc
        persist_dir = str(self.settings.vectorstore.persist_directory)
        logger.info("Building Chroma index", extra={"extra": {"documents": len(documents)}})
        store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.settings.vectorstore.collection_name,
            persist_directory=persist_dir,
        )
        return store

    def _load_chroma(self) -> Any:
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError as exc:
            raise DependencyNotInstalledError("Install chromadb and langchain-community.") from exc
        return Chroma(
            collection_name=self.settings.vectorstore.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.settings.vectorstore.persist_directory),
        )

    def _build_faiss(self, documents: list[Any]) -> Any:
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise DependencyNotInstalledError("Install faiss-cpu and langchain-community.") from exc
        index_dir = Path(self.settings.paths.index_dir) / self.settings.vectorstore.faiss_index_name
        logger.info("Building FAISS index", extra={"extra": {"documents": len(documents)}})
        store = FAISS.from_documents(documents, self.embeddings)
        store.save_local(str(index_dir))
        return store

    def _load_faiss(self) -> Any:
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise DependencyNotInstalledError("Install faiss-cpu and langchain-community.") from exc
        index_dir = Path(self.settings.paths.index_dir) / self.settings.vectorstore.faiss_index_name
        return FAISS.load_local(
            str(index_dir),
            self.embeddings,
            allow_dangerous_deserialization=(
                self.settings.vectorstore.allow_dangerous_deserialization
            ),
        )
