"""PDF loading, multimodal extraction, splitting, and context formatting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import DocumentProcessingSection
from core.exceptions import DependencyNotInstalledError, IngestionError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MultimodalElements:
    """Separated multimodal elements from a PDF."""

    texts: list[Any]
    tables: list[Any]
    images_base64: list[str]


def load_pdf_pages(pdf_path: str | Path) -> list[Any]:
    """Load a PDF as page-level LangChain documents using PyPDFLoader."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as exc:
        raise DependencyNotInstalledError("Install langchain-community and pypdf.") from exc

    path = Path(pdf_path)
    try:
        loader = PyPDFLoader(str(path))
        documents = loader.load()
        for doc in documents:
            doc.metadata.update(
                {"source_file": path.name, "source_path": str(path), "file_type": "pdf"}
            )
        return documents
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Failed to load PDF {path}: {exc}") from exc


def split_documents(
    documents: list[Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Any]:
    """Split documents into chunks for retrieval."""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise DependencyNotInstalledError("Install langchain to split documents.") from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Split documents",
        extra={"extra": {"input_documents": len(documents), "chunks": len(chunks)}},
    )
    return chunks


def partition_multimodal_pdf(
    pdf_path: str | Path,
    config: DocumentProcessingSection,
) -> MultimodalElements:
    """Extract text chunks, tables, and image payloads using unstructured.

    Args:
        pdf_path: Path to the PDF.
        config: Processing configuration.

    Returns:
        Separated text, table, and image elements.
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        raise DependencyNotInstalledError("Install unstructured[all-docs] for multimodal PDFs.") from exc

    path = Path(pdf_path)
    try:
        chunks = partition_pdf(
            filename=str(path),
            infer_table_structure=config.infer_table_structure,
            strategy=config.unstructured_strategy,
            extract_image_block_types=config.extract_image_block_types,
            extract_image_block_to_payload=True,
            chunking_strategy=config.chunking_strategy,
            max_characters=config.max_characters,
            combine_text_under_n_chars=config.combine_text_under_n_chars,
            new_after_n_chars=config.new_after_n_chars,
        )
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Failed to partition multimodal PDF {path}: {exc}") from exc

    elements = separate_multimodal_elements(chunks)
    logger.info(
        "Partitioned multimodal PDF",
        extra={
            "extra": {
                "pdf": str(path),
                "texts": len(elements.texts),
                "tables": len(elements.tables),
                "images": len(elements.images_base64),
            }
        },
    )
    return elements


def separate_multimodal_elements(chunks: list[Any]) -> MultimodalElements:
    """Separate unstructured CompositeElement chunks into text, table, and image lists."""
    texts: list[Any] = []
    tables: list[Any] = []
    images_base64: list[str] = []

    for chunk in chunks:
        if "CompositeElement" not in str(type(chunk)):
            continue
        texts.append(chunk)
        for element in getattr(getattr(chunk, "metadata", None), "orig_elements", []) or []:
            element_type = str(type(element))
            if "Table" in element_type:
                tables.append(element)
            elif "Image" in element_type:
                image = getattr(getattr(element, "metadata", None), "image_base64", None)
                if image:
                    images_base64.append(image)

    return MultimodalElements(texts=texts, tables=tables, images_base64=images_base64)


def format_context(
    retrieved_docs: list[Any],
    include_tables: bool = True,
    include_images_as_text: bool = True,
    max_original_chars: int = 1200,
) -> str:
    """Format retrieved LangChain documents into a compact text context."""
    lines: list[str] = []
    for index, doc in enumerate(retrieved_docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        modality = metadata.get("modality", "text")
        content = getattr(doc, "page_content", str(doc))

        if modality == "text":
            original = metadata.get("original", "")
            snippet = str(original)[:max_original_chars]
            if snippet and snippet != content:
                lines.append(f"[{index}] TEXT — Summary: {content}\nOriginal snippet: {snippet}")
            else:
                lines.append(f"[{index}] TEXT — Summary: {content}")
        elif modality == "table" and include_tables:
            lines.append(f"[{index}] TABLE — Summary: {content}")
        elif modality == "image" and include_images_as_text:
            lines.append(f"[{index}] IMAGE — Summary: {content}")

    return "\n\n".join(lines)
