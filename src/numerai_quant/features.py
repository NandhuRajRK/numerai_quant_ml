"""Feature detection, selection, and prediction normalization utilities."""

from __future__ import annotations

import importlib
import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def detect_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return Numerai feature columns in stable order."""
    features = [column for column in df.columns if column.startswith("feature_")]
    if not features:
        raise ValueError("No feature columns found. Expected columns beginning with 'feature_'.")
    return sorted(features)


def prune_correlated_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    max_abs_corr: float = 0.995,
    sample_size: int | None = 25_000,
) -> tuple[list[str], dict[str, float | int]]:
    """Drop highly correlated features using a simple upper-triangle filter."""
    if len(feature_cols) <= 1:
        return feature_cols, {"num_input_features": len(feature_cols), "num_dropped": 0}

    working = df[feature_cols]
    if sample_size is not None and len(working) > sample_size:
        working = working.sample(n=sample_size, random_state=42)

    corr = working.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if (upper[column] > max_abs_corr).any()]
    kept = [column for column in feature_cols if column not in set(to_drop)]
    return kept, {
        "num_input_features": len(feature_cols),
        "num_dropped": len(to_drop),
        "max_abs_corr_threshold": float(max_abs_corr),
    }


def top_importance_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "target",
    top_n: int = 1000,
    selector_params: dict[str, object] | None = None,
) -> tuple[list[str], pd.Series]:
    """Fit a lightweight LightGBM selector and keep the top-N important features."""
    if top_n >= len(feature_cols):
        return feature_cols, pd.Series(dtype=float)

    module = importlib.import_module("lightgbm")
    selector_cls = module.LGBMRegressor
    params = {
        "objective": "regression",
        "metric": "l2",
        "n_estimators": 250,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "random_state": 42,
        "n_jobs": 6,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if selector_params:
        params.update(selector_params)

    selector = selector_cls(**params)
    selector.fit(df[feature_cols], df[target_col])
    importances = pd.Series(selector.feature_importances_, index=feature_cols, dtype=float)
    importances = importances.sort_values(ascending=False)
    if importances.sum() <= 0:
        return feature_cols[:top_n], importances
    selected = list(importances.head(top_n).index)
    return selected, importances


def apply_feature_controls(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "target",
    config: dict[str, object] | None = None,
) -> tuple[list[str], dict[str, object]]:
    """Apply correlation pruning and importance filtering from config."""
    controls = config or {}
    selected = list(feature_cols)
    summary: dict[str, object] = {
        "num_detected_features": len(feature_cols),
        "num_after_correlation_pruning": len(feature_cols),
        "num_after_importance_filtering": len(feature_cols),
        "correlation_pruning_enabled": False,
        "importance_filtering_enabled": False,
    }

    corr_cfg = controls.get("correlation_pruning", {})
    if isinstance(corr_cfg, dict) and corr_cfg.get("enabled", False):
        selected, corr_summary = prune_correlated_features(
            df,
            selected,
            max_abs_corr=float(corr_cfg.get("max_abs_corr", 0.995)),
            sample_size=int(corr_cfg["sample_size"]) if corr_cfg.get("sample_size") else None,
        )
        summary["correlation_pruning_enabled"] = True
        summary["num_after_correlation_pruning"] = len(selected)
        summary["num_pruned_correlated"] = int(corr_summary["num_dropped"])

    importance_cfg = controls.get("importance_filtering", {})
    if isinstance(importance_cfg, dict) and importance_cfg.get("enabled", False):
        selected, importances = top_importance_features(
            df,
            selected,
            target_col=target_col,
            top_n=int(importance_cfg.get("top_n", len(selected))),
            selector_params=dict(importance_cfg.get("selector_params", {})),
        )
        summary["importance_filtering_enabled"] = True
        summary["num_after_importance_filtering"] = len(selected)
        summary["top_importance_features"] = list(importances.head(20).index)

    LOGGER.info(
        "Feature controls | detected=%s after_corr=%s after_importance=%s",
        summary["num_detected_features"],
        summary["num_after_correlation_pruning"],
        summary["num_after_importance_filtering"],
    )
    return selected, summary


def ensure_identifier_column(df: pd.DataFrame, *, id_col: str = "id") -> pd.DataFrame:
    """Ensure an identifier column exists, promoting the index when needed."""
    if id_col in df.columns:
        return df
    if df.index.name == id_col:
        return df.reset_index()
    return df.assign(**{id_col: df.index.astype(str)})


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
