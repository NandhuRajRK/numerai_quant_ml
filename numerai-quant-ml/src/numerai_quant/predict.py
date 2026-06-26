"""Prediction file helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from numerai_quant.features import validate_prediction_frame


def save_predictions(frame: pd.DataFrame, path: Path) -> Path:
    """Save predictions as CSV or parquet based on the file suffix."""
    validate_prediction_frame(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def read_predictions(path: Path) -> pd.DataFrame:
    """Read predictions from CSV or parquet."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
