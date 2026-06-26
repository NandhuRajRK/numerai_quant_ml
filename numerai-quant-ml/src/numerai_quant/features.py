"""Feature detection and prediction normalization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return Numerai feature columns in stable order."""
    features = [column for column in df.columns if column.startswith("feature_")]
    if not features:
        raise ValueError("No feature columns found. Expected columns beginning with 'feature_'.")
    return sorted(features)


def rank_normalize(values: pd.Series | np.ndarray) -> pd.Series:
    """Convert arbitrary prediction values to percentile ranks in [0, 1]."""
    series = pd.Series(values, copy=False)
    if series.empty:
        return series.astype(float)
    return series.rank(method="first", pct=True).clip(0.0, 1.0).astype(float)


def neutralize_series(
    predictions: pd.Series | np.ndarray,
    exposures: pd.DataFrame,
    *,
    proportion: float = 0.5,
) -> pd.Series:
    """Neutralize predictions against a set of exposure columns."""
    prediction_series = pd.Series(predictions, copy=False).astype(float)
    if exposures.empty or prediction_series.empty:
        return rank_normalize(prediction_series)

    centered_predictions = prediction_series - prediction_series.mean()
    centered_exposures = exposures.astype(float) - exposures.astype(float).mean(axis=0)
    coefficients, *_ = np.linalg.lstsq(
        centered_exposures.to_numpy(),
        centered_predictions.to_numpy(),
        rcond=None,
    )
    neutralized = centered_predictions.to_numpy() - (
        proportion * centered_exposures.to_numpy().dot(coefficients)
    )
    return rank_normalize(neutralized)


def make_prediction_frame(
    ids: pd.Series,
    predictions: pd.Series | np.ndarray,
    *,
    id_col: str = "id",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Build a Numerai-compatible prediction frame."""
    normalized = rank_normalize(predictions)
    frame = pd.DataFrame({id_col: ids.to_numpy(), prediction_col: normalized.to_numpy()})
    validate_prediction_frame(frame, id_col=id_col, prediction_col=prediction_col)
    return frame


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    id_col: str = "id",
    prediction_col: str = "prediction",
) -> None:
    """Validate required submission columns and prediction value range."""
    missing = {id_col, prediction_col} - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame missing required columns: {sorted(missing)}")
    if frame[id_col].isna().any():
        raise ValueError("Prediction frame contains missing ids.")
    if frame[prediction_col].isna().any():
        raise ValueError("Prediction frame contains missing predictions.")
    if not frame[prediction_col].between(0.0, 1.0).all():
        raise ValueError("Predictions must be between 0 and 1.")
