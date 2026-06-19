"""Embedding model factory functions."""

from __future__ import annotations

import os
from typing import Any

from core.config import EmbeddingsSection
from core.exceptions import ConfigurationError, DependencyNotInstalledError


def create_embeddings(config: EmbeddingsSection) -> Any:
    """Create a LangChain-compatible embeddings object."""
    if config.provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise DependencyNotInstalledError("Install langchain-openai to use OpenAI embeddings.") from exc
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ConfigurationError(f"Missing API key. Set environment variable {config.api_key_env}.")
        return OpenAIEmbeddings(model=config.model_name, api_key=api_key)

    if config.provider == "huggingface":
        try:
            import torch
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise DependencyNotInstalledError(
                "Install sentence-transformers and langchain-community to use HuggingFace embeddings."
            ) from exc
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(
            model_name=config.model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": config.normalize_embeddings},
        )

    raise ConfigurationError(f"Unsupported embedding provider: {config.provider}")
