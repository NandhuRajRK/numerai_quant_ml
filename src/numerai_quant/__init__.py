"""Numerai Tournament quant ML baseline package."""

from __future__ import annotations

import os


def _configure_headless_matplotlib() -> None:
    """Prefer a non-interactive backend in notebook and batch environments."""
    backend = os.environ.get("MPLBACKEND", "")
    if backend.startswith("module://matplotlib_inline"):
        os.environ["MPLBACKEND"] = "Agg"


_configure_headless_matplotlib()

__version__ = "0.1.0"
