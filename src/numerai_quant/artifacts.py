"""Artifact tracking helpers for research runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from numerai_quant.utils import ensure_directories


def init_run_directory(root: Path, run_name: str) -> Path:
    """Create and return a run directory."""
    run_dir = root / run_name
    ensure_directories([run_dir, run_dir / "plots", run_dir / "models", run_dir / "checkpoints"])
    return run_dir


def save_json(data: dict[str, Any], path: Path) -> Path:
    """Save a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_yaml(data: dict[str, Any], path: Path) -> Path:
    """Save a YAML artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return path


def save_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    """Save a dataframe to csv or parquet based on suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path
