"""CLI entry point for document ingestion."""

from __future__ import annotations

import argparse
import logging

from core.config import load_settings
from core.logging import setup_logging
from pipelines.ingestion import IngestionPipeline

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG vector store.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--pdf", action="append", default=[], help="Path to a PDF file. Repeatable.")
    parser.add_argument("--pdf-dir", action="append", default=[], help="Directory of PDF files. Repeatable.")
    return parser.parse_args()


def main() -> None:
    """Run ingestion."""
    args = parse_args()
    settings = load_settings(args.config)
    setup_logging(settings.logging.level, settings.logging.json_logs, settings.app.verbose)
    paths = [*args.pdf, *args.pdf_dir]
    if not paths:
        paths = [str(settings.paths.raw_data_dir)]
    IngestionPipeline(settings).ingest(paths)
    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()
