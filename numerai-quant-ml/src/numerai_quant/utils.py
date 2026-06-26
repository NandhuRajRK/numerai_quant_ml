"""Shared utilities for paths, logging, and environment handling."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    """Configure readable application logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_directories(paths: list[Path]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def require_columns(columns: set[str], required: set[str], context: str) -> None:
    """Raise a helpful error if required columns are missing."""
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def timestamp_slug() -> str:
    """Return a compact UTC timestamp for artifact names."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
