"""Document ingestion pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.config import Settings
from core.exceptions import IngestionError
from modules.cache import DiskJsonCache
from modules.document_processing import load_pdf_pages, partition_multimodal_pdf, split_documents
from modules.summarizers import ImageSummarizer, TextTableSummarizer, build_summary_documents
from modules.vectorstores import VectorStoreManager
from utils.file_utils import iter_pdf_files

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Production ingestion pipeline for text-only and multimodal PDFs."""

    def __init__(self, settings: Settings):
        """Initialize pipeline."""
        self.settings = settings
        self.cache = DiskJsonCache(
            settings.paths.cache_dir,
            enabled=settings.cache.enabled,
            ttl_seconds=settings.cache.ttl_seconds,
        )

    def ingest(self, pdf_paths: list[str | Path]) -> Any:
        """Ingest PDFs and build a vector index.

        Args:
            pdf_paths: File or directory paths containing PDFs.

        Returns:
            Persisted vector store instance.
        """
        pdf_files = self._resolve_pdf_files(pdf_paths)
        if not pdf_files:
            raise IngestionError("No PDF files found for ingestion.")

        if self.settings.document_processing.mode == "text":
            documents = self._ingest_text_pdfs(pdf_files)
        else:
            documents = self._ingest_multimodal_pdfs(pdf_files)

        manager = VectorStoreManager(self.settings)
        return manager.build(documents)

    def _resolve_pdf_files(self, paths: list[str | Path]) -> list[Path]:
        pdf_files: list[Path] = []
        for path in paths:
            pdf_files.extend(iter_pdf_files(path))
        unique = sorted({file.resolve() for file in pdf_files})
        logger.info("Resolved PDF files", extra={"extra": {"count": len(unique)}})
        return unique

    def _ingest_text_pdfs(self, pdf_files: list[Path]) -> list[Any]:
        pages: list[Any] = []
        for pdf in pdf_files:
            pages.extend(load_pdf_pages(pdf))
        return split_documents(
            pages,
            chunk_size=self.settings.document_processing.chunk_size,
            chunk_overlap=self.settings.document_processing.chunk_overlap,
        )

    def _ingest_multimodal_pdfs(self, pdf_files: list[Path]) -> list[Any]:
        all_documents: list[Any] = []
        text_summarizer = TextTableSummarizer(self.settings, cache=self.cache)
        image_summarizer = ImageSummarizer(self.settings, cache=self.cache)

        for pdf in pdf_files:
            elements = partition_multimodal_pdf(pdf, self.settings.document_processing)
            table_html = [getattr(getattr(table, "metadata", None), "text_as_html", str(table)) for table in elements.tables]
            text_summaries = text_summarizer.summarize_batch(elements.texts, max_concurrency=1)
            table_summaries = text_summarizer.summarize_batch(table_html, max_concurrency=1)
            image_summaries = image_summarizer.summarize_batch(elements.images_base64, max_concurrency=1)
            documents = build_summary_documents(
                texts=elements.texts,
                tables=elements.tables,
                images_base64=elements.images_base64,
                text_summaries=text_summaries,
                table_summaries=table_summaries,
                image_summaries=image_summaries,
            )
            for document in documents:
                document.metadata.update({"source_file": pdf.name, "source_path": str(pdf)})
            all_documents.extend(documents)

        logger.info(
            "Built multimodal summary documents",
            extra={"extra": {"documents": len(all_documents)}},
        )
        return all_documents
