"""LLM factory functions."""

from __future__ import annotations

import os
from typing import Any

from core.config import LLMSection
from core.exceptions import ConfigurationError, DependencyNotInstalledError


def _require_api_key(config: LLMSection) -> str | None:
    """Validate and return an API key when the provider requires one."""
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise ConfigurationError(
            f"Missing API key. Set environment variable {config.api_key_env} "
            f"for provider '{config.provider}'."
        )
    return api_key


def create_chat_model(config: LLMSection) -> Any:
    """Create a LangChain-compatible chat model.

    Args:
        config: Chat model configuration.

    Returns:
        A LangChain chat model instance.
    """
    if config.provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise DependencyNotInstalledError("Install langchain-groq to use Groq models.") from exc
        return ChatGroq(
            groq_api_key=_require_api_key(config),
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if config.provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise DependencyNotInstalledError("Install langchain-openai to use OpenAI models.") from exc
        return ChatOpenAI(
            api_key=_require_api_key(config),
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    raise ConfigurationError(f"Unsupported LLM provider: {config.provider}")
