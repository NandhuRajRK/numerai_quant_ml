"""Validation diagnostics and walk-forward evaluation for Numerai predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from numerai_quant.features import neutralize_series
from numerai_quant.metrics import era_correlations, summarize_correlations

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardFold:
    """Definition of a single walk-forward fold."""

    fold_number: int
    train_eras: list[str]
    validation_eras: list[str]


def ordered_eras(frame: pd.DataFrame, era_col: str = "era") -> list[str]:
    """Return eras in first-seen order."""
    return frame[era_col].drop_duplicates().astype(str).tolist()


def build_walk_forward_folds(
    frame: pd.DataFrame,
    *,
    era_col: str = "era",
    min_train_eras: int = 24,
    validation_eras: int = 4,
    embargo_eras: int = 1,
    step_size: int = 4,
    max_folds: int | None = None,
) -> list[WalkForwardFold]:
    """Build walk-forward folds using ordered eras."""
    eras = ordered_eras(frame, era_col=era_col)
    if len(eras) < min_train_eras + validation_eras:
        raise ValueError("Not enough eras for the requested walk-forward configuration.")

    folds: list[WalkForwardFold] = []
    fold_number = 1
    for validation_start in range(min_train_eras + embargo_eras, len(eras), step_size):
        validation_end = validation_start + validation_eras
        if validation_end > len(eras):
            break
        train_end = validation_start - embargo_eras
        folds.append(
            WalkForwardFold(
                fold_number=fold_number,
                train_eras=eras[:train_end],
                validation_eras=eras[validation_start:validation_end],
            )
        )
        fold_number += 1

    if max_folds is not None and len(folds) > max_folds:
        folds = folds[-max_folds:]
    return [
        WalkForwardFold(
            fold_number=index,
            train_eras=fold.train_eras,
            validation_eras=fold.validation_eras,
        )
        for index, fold in enumerate(folds, start=1)
    ]


def feature_exposure(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    prediction_col: str = "prediction",
    sample_size: int | None = 500,
) -> pd.Series:
    """Estimate absolute correlation between predictions and feature columns."""
    selected_features = feature_cols[:sample_size] if sample_size else feature_cols
    exposures: dict[str, float] = {}
    prediction_rank = frame[prediction_col].rank(method="average")
    if prediction_rank.nunique(dropna=False) <= 1:
        return pd.Series({feature: 0.0 for feature in selected_features}).sort_values(
            ascending=False
        )

    for feature in selected_features:
        feature_rank = frame[feature].rank(method="average")
        if feature_rank.nunique(dropna=False) <= 1:
            exposures[feature] = 0.0
            continue
        corr = prediction_rank.corr(feature_rank, method="pearson")
        exposures[feature] = 0.0 if pd.isna(corr) else abs(float(corr))

    return pd.Series(exposures).sort_values(ascending=False)


def apply_feature_neutralization(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    prediction_col: str = "prediction",
    sample_size: int = 50,
    proportion: float = 0.5,
) -> pd.Series:
    """Neutralize predictions against the most exposed features."""
    exposures = feature_exposure(
        frame,
        feature_cols,
        prediction_col=prediction_col,
        sample_size=len(feature_cols),
    )
    selected = list(exposures.head(sample_size).index)
    return neutralize_series(
        frame[prediction_col],
        frame[selected],
        proportion=proportion,
    )


def validation_report(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    era_col: str = "era",
    target_col: str = "target",
    prediction_col: str = "prediction",
    exposure_sample_size: int | None = 500,
) -> dict[str, Any]:
    """Build era-wise validation and feature exposure diagnostics."""
    LOGGER.warning(
        "Validation is era-wise, but it is still a backtest. "
        "Avoid feature, target, and era leakage."
    )
    LOGGER.warning("Do not tune repeatedly on validation eras without a holdout plan.")

    correlations = era_correlations(
        frame,
        era_col=era_col,
        target_col=target_col,
        prediction_col=prediction_col,
    )
    exposures = feature_exposure(
        frame,
        feature_cols,
        prediction_col=prediction_col,
        sample_size=exposure_sample_size,
    )

    summary = summarize_correlations(correlations)
    summary["mean_abs_feature_exposure"] = float(np.mean(exposures)) if len(exposures) else 0.0
    summary["max_abs_feature_exposure"] = float(exposures.max()) if len(exposures) else 0.0
    summary["num_eras"] = int(correlations.shape[0])

    return {
        "summary": summary,
        "era_correlations": correlations,
        "feature_exposure": exposures,
    }


def summarize_fold_result(
    fold_number: int,
    model_name: str,
    report: dict[str, Any],
) -> dict[str, float | int | str]:
    """Convert a validation report into a flat fold metric row."""
    summary = report["summary"]
    return {
        "fold": fold_number,
        "model_name": model_name,
        "mean_corr": float(summary["mean_corr"]),
        "std_corr": float(summary["std_corr"]),
        "sharpe_like": float(summary["sharpe_like"]),
        "max_drawdown": float(summary["max_drawdown"]),
        "mean_abs_feature_exposure": float(summary["mean_abs_feature_exposure"]),
        "max_abs_feature_exposure": float(summary["max_abs_feature_exposure"]),
        "num_eras": int(summary["num_eras"]),
    }
