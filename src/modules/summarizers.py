"""Text, table, and image summarization modules."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from core.config import Settings
from core.exceptions import DependencyNotInstalledError, GenerationError
from modules.cache import DiskJsonCache
from modules.llms import create_chat_model
from utils.hashing import sha256_text

logger = logging.getLogger(__name__)

TEXT_TABLE_PROMPT = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.
Respond only with the summary, without preamble.

Table or text chunk:
{element}
""".strip()

IMAGE_PROMPT = """
Describe the image in detail for use in a retrieval system.
Be specific about diagrams, charts, axes, tables, algorithms, and visual evidence.
""".strip()


class TextTableSummarizer:
    """Summarize text chunks and HTML tables."""

    def __init__(self, settings: Settings, cache: DiskJsonCache | None = None):
        """Initialize summarizer."""
        self.settings = settings
        self.cache = cache
        self.llm = create_chat_model(settings.llm)
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError as exc:
            raise DependencyNotInstalledError("Install langchain-core.") from exc
        prompt = ChatPromptTemplate.from_template(TEXT_TABLE_PROMPT)
        self.chain = {"element": lambda value: value} | prompt | self.llm | StrOutputParser()

    def summarize_batch(self, elements: list[Any], max_concurrency: int = 1) -> list[str]:
        """Summarize a batch of elements with caching."""
        summaries: list[str] = []
        uncached: list[str] = []
        uncached_positions: list[int] = []
        summaries = [""] * len(elements)

        for position, element in enumerate(elements):
            text = _element_to_text(element)
            cache_key = f"text-table-summary:{sha256_text(text)}"
            cached = self.cache.get(cache_key) if self.cache else None
            if cached is not None:
                summaries[position] = str(cached)
            else:
                uncached.append(text)
                uncached_positions.append(position)

        if uncached:
            try:
                generated = self.chain.batch(uncached, {"max_concurrency": max_concurrency})
            except Exception as exc:  # noqa: BLE001
                raise GenerationError(f"Text/table summarization failed: {exc}") from exc
            for position, text, summary in zip(uncached_positions, uncached, generated, strict=True):
                summaries[position] = str(summary)
                if self.cache:
                    self.cache.set(f"text-table-summary:{sha256_text(text)}", str(summary))

        return summaries


class ImageSummarizer:
    """Summarize base64 encoded images using a vision-capable chat model."""

    def __init__(self, settings: Settings, cache: DiskJsonCache | None = None):
        """Initialize image summarizer."""
        self.settings = settings
        self.cache = cache
        self.llm = create_chat_model(settings.vision_llm)
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError as exc:
            raise DependencyNotInstalledError("Install langchain-core.") from exc

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": IMAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
                ],
            )
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        self.chain = prompt | self.llm | StrOutputParser()

    def summarize_batch(self, images_base64: list[str], max_concurrency: int = 1) -> list[str]:
        """Summarize image payloads with caching."""
        summaries = [""] * len(images_base64)
        uncached: list[str] = []
        uncached_positions: list[int] = []
        for position, image in enumerate(images_base64):
            cache_key = f"image-summary:{sha256_text(image)}"
            cached = self.cache.get(cache_key) if self.cache else None
            if cached is not None:
                summaries[position] = str(cached)
            else:
                uncached.append(image)
                uncached_positions.append(position)

        if uncached:
            try:
                generated = self.chain.batch(uncached, {"max_concurrency": max_concurrency})
            except Exception as exc:  # noqa: BLE001
                raise GenerationError(f"Image summarization failed: {exc}") from exc
            for position, image, summary in zip(uncached_positions, uncached, generated, strict=True):
                summaries[position] = str(summary)
                if self.cache:
                    self.cache.set(f"image-summary:{sha256_text(image)}", str(summary))
        return summaries


def build_summary_documents(
    texts: list[Any],
    tables: list[Any],
    images_base64: list[str],
    text_summaries: list[str],
    table_summaries: list[str],
    image_summaries: list[str],
) -> list[Any]:
    """Build flat retrieval documents from modality-specific summaries."""
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise DependencyNotInstalledError("Install langchain-core.") from exc

    documents: list[Any] = []
    for original, summary in zip(texts, text_summaries, strict=False):
        documents.append(
            Document(
                page_content=summary,
                metadata={
                    "id": str(uuid.uuid4()),
                    "modality": "text",
                    "original": _element_to_text(original),
                },
            )
        )

    for original, summary in zip(tables, table_summaries, strict=False):
        documents.append(
            Document(
                page_content=summary,
                metadata={
                    "id": str(uuid.uuid4()),
                    "modality": "table",
                    "original": _table_to_html(original),
                },
            )
        )

    for image, summary in zip(images_base64, image_summaries, strict=False):
        documents.append(
            Document(
                page_content=summary,
                metadata={"id": str(uuid.uuid4()), "modality": "image", "image_b64": image},
            )
        )

    return documents


def _element_to_text(element: Any) -> str:
    """Extract text from an arbitrary unstructured or LangChain element."""
    if hasattr(element, "page_content"):
        return str(element.page_content)
    if hasattr(element, "text"):
        return str(element.text)
    return str(element)


def _table_to_html(table: Any) -> str:
    """Extract HTML from an unstructured table when available."""
    metadata = getattr(table, "metadata", None)
    html = getattr(metadata, "text_as_html", None) if metadata else None
    return str(html or _element_to_text(table))
