"""Lightweight JSON disk cache."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from utils.hashing import sha256_text


class DiskJsonCache:
    """Simple file-backed JSON cache with optional TTL."""

    def __init__(self, cache_dir: str | Path, enabled: bool = True, ttl_seconds: int | None = None):
        """Initialize the cache.

        Args:
            cache_dir: Directory where cache files are stored.
            enabled: Disable reads/writes when false.
            ttl_seconds: Optional time-to-live in seconds.
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{sha256_text(key)}.json"

    def get(self, key: str) -> Any | None:
        """Return cached value or None when missing/expired."""
        if not self.enabled:
            return None
        path = self._path_for_key(key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        created_at = float(payload.get("created_at", 0))
        if self.ttl_seconds is not None and time.time() - created_at > self.ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        return payload.get("value")

    def set(self, key: str, value: Any) -> None:
        """Write a cached value."""
        if not self.enabled:
            return
        path = self._path_for_key(key)
        payload = {"created_at": time.time(), "value": value}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
