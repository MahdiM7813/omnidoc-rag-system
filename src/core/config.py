"""Configuration loading and validation utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from core.exceptions import ConfigurationError


class AppSection(BaseModel):
    """Application metadata and runtime switches."""

    name: str = "production-multimodal-rag"
    environment: str = "development"
    debug: bool = False
    verbose: bool = False


class PathsSection(BaseModel):
    """Filesystem paths used by the application."""

    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    index_dir: Path = Path("data/indexes")
    cache_dir: Path = Path("data/cache")


class LoggingSection(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    json_logs: bool = True

    @field_validator("level")
    @classmethod
    def uppercase_level(cls, value: str) -> str:
        """Normalize logging levels to uppercase."""
        return value.upper()


class LLMSection(BaseModel):
    """Chat model configuration."""

    provider: Literal["groq", "openai"] = "groq"
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.2
    max_tokens: int = 1024
    api_key_env: str = "GROQ_API_KEY"


class EmbeddingsSection(BaseModel):
    """Embedding model configuration."""

    provider: Literal["openai", "huggingface"] = "openai"
    model_name: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"
    device: str = "auto"
    normalize_embeddings: bool = True


class VectorStoreSection(BaseModel):
    """Vector store configuration."""

    provider: Literal["chroma", "faiss"] = "chroma"
    collection_name: str = "multimodal_rag"
    persist_directory: Path = Path("data/indexes/chroma")
    faiss_index_name: str = "faiss_index"
    allow_dangerous_deserialization: bool = False


class DocumentProcessingSection(BaseModel):
    """Document loading and chunking configuration."""

    mode: Literal["text", "multimodal"] = "multimodal"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    unstructured_strategy: str = "hi_res"
    infer_table_structure: bool = True
    extract_image_block_types: list[str] = Field(default_factory=lambda: ["Image", "Table"])
    chunking_strategy: str = "by_title"
    max_characters: int = 10000
    combine_text_under_n_chars: int = 2000
    new_after_n_chars: int = 6000
    max_original_chars: int = 1200


class RetrievalSection(BaseModel):
    """Retriever configuration."""

    top_k: int = 6
    score_threshold: float | None = None


class ConversationSection(BaseModel):
    """Conversation memory configuration."""

    enabled: bool = True
    max_turns: int = 8
    long_term_memory_path: Path = Path("data/cache/conversation_memory.json")


class CacheSection(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    ttl_seconds: int = 86400


class RagSection(BaseModel):
    """RAG prompt and response configuration."""

    system_prompt: str
    include_tables: bool = True
    include_images_as_text: bool = True
    return_sources: bool = True


class EvaluationSection(BaseModel):
    """Evaluation configuration."""

    questions_path: Path = Path("configs/eval_questions.yaml")
    output_path: Path = Path("data/processed/eval_results.jsonl")


class Settings(BaseModel):
    """Top-level application settings."""

    app: AppSection = Field(default_factory=AppSection)
    paths: PathsSection = Field(default_factory=PathsSection)
    logging: LoggingSection = Field(default_factory=LoggingSection)
    llm: LLMSection = Field(default_factory=LLMSection)
    vision_llm: LLMSection = Field(
        default_factory=lambda: LLMSection(provider="openai", model_name="gpt-4o-mini")
    )
    embeddings: EmbeddingsSection = Field(default_factory=EmbeddingsSection)
    vectorstore: VectorStoreSection = Field(default_factory=VectorStoreSection)
    document_processing: DocumentProcessingSection = Field(default_factory=DocumentProcessingSection)
    retrieval: RetrievalSection = Field(default_factory=RetrievalSection)
    conversation: ConversationSection = Field(default_factory=ConversationSection)
    cache: CacheSection = Field(default_factory=CacheSection)
    rag: RagSection = Field(
        default_factory=lambda: RagSection(
            system_prompt=(
                "You are a reliable research assistant. Answer strictly from the provided "
                "context. If the answer is not in the context, say I don't know."
            )
        )
    )
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not exist."""
        for path in [
            self.paths.raw_data_dir,
            self.paths.processed_data_dir,
            self.paths.index_dir,
            self.paths.cache_dir,
            self.vectorstore.persist_directory,
            self.conversation.long_term_memory_path.parent,
            self.evaluation.output_path.parent,
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in YAML values."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def load_settings(config_path: str | Path = "configs/default.yaml") -> Settings:
    """Load and validate application settings.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        Validated settings object.

    Raises:
        ConfigurationError: If the configuration file is missing or invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        settings = Settings.model_validate(_expand_env_vars(data))
    except Exception as exc:  # noqa: BLE001 - convert library errors to domain error
        raise ConfigurationError(f"Invalid configuration {path}: {exc}") from exc

    settings.ensure_directories()
    return settings
