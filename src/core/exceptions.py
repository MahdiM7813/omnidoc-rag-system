"""Custom exception hierarchy for the RAG system."""


class RagError(Exception):
    """Base class for all application-specific exceptions."""


class ConfigurationError(RagError):
    """Raised when configuration is invalid or incomplete."""


class DependencyNotInstalledError(RagError):
    """Raised when an optional dependency required for a feature is missing."""


class IngestionError(RagError):
    """Raised when document ingestion fails."""


class RetrievalError(RagError):
    """Raised when retrieval fails."""


class GenerationError(RagError):
    """Raised when answer generation fails."""
