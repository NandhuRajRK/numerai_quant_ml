"""Metrics for Numerai-style era validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def spearman_corr(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Compute Spearman correlation via rank correlation."""
    true_rank = pd.Series(y_true).rank(method="average")
    pred_rank = pd.Series(y_pred).rank(method="average")
    corr = true_rank.corr(pred_rank, method="pearson")
    return float(0.0 if pd.isna(corr) else corr)


def era_correlations(
    frame: pd.DataFrame,
    *,
    era_col: str = "era",
    target_col: str = "target",
    prediction_col: str = "prediction",
) -> pd.Series:
    """Compute Spearman correlation for each era."""
    required = {era_col, target_col, prediction_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns for era correlations: {sorted(missing)}")

    return frame.groupby(era_col, sort=False).apply(
        lambda era: spearman_corr(era[target_col], era[prediction_col]),
        include_groups=False,
    )


def sharpe_like(correlations: pd.Series) -> float:
    """Return mean era correlation divided by standard deviation."""
    std = correlations.std(ddof=0)
    if std == 0 or pd.isna(std):
        return 0.0
    return float(correlations.mean() / std)


def max_drawdown(values: pd.Series) -> float:
    """Compute max drawdown on cumulative era correlations."""
    cumulative = values.cumsum()
    running_max = cumulative.cummax()
    drawdowns = cumulative - running_max
    return float(drawdowns.min())


def summarize_correlations(correlations: pd.Series) -> dict[str, float]:
    """Summarize era correlations with common diagnostics."""
    return {
        "mean_corr": float(correlations.mean()),
        "std_corr": float(correlations.std(ddof=0)),
        "sharpe_like": sharpe_like(correlations),
        "max_drawdown": max_drawdown(correlations),
        "min_corr": float(correlations.min()),
        "max_corr": float(correlations.max()),
    }
