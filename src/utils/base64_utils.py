"""Base64 helper functions."""

from __future__ import annotations

import base64
from pathlib import Path


def image_file_to_base64(path: str | Path) -> str:
    """Encode an image file as base64 text."""
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def base64_to_bytes(value: str) -> bytes:
    """Decode base64 text into bytes."""
    return base64.b64decode(value)
