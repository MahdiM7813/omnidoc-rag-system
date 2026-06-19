"""Filesystem utilities."""

from __future__ import annotations

import re
from pathlib import Path


PDF_PATTERN = "*.pdf"


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def iter_pdf_files(path: str | Path) -> list[Path]:
    """Return sorted PDF files from a file or directory.

    Args:
        path: A PDF file or a directory containing PDFs.

    Returns:
        Sorted list of PDF paths.
    """
    resolved = Path(path)
    if resolved.is_file() and resolved.suffix.lower() == ".pdf":
        return [resolved]
    if resolved.is_dir():
        return sorted(resolved.rglob(PDF_PATTERN))
    return []


def safe_filename(value: str, max_length: int = 120) -> str:
    """Convert arbitrary text into a safe filename stem."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    return cleaned[:max_length] or "file"
