"""Plotting and report generation for Numerai experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def plot_cumulative_correlation(era_corr: pd.Series, path: Path) -> Path:
    """Save a cumulative correlation chart."""
    figure, axis = plt.subplots(figsize=(10, 4))
    era_corr.cumsum().plot(ax=axis, linewidth=2)
    axis.set_title("Cumulative Era Correlation")
    axis.set_xlabel("Era")
    axis.set_ylabel("Cumulative CORR")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_correlation_histogram(era_corr: pd.Series, path: Path) -> Path:
    """Save a histogram of era correlations."""
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(era_corr.to_numpy(), bins=25, edgecolor="black", alpha=0.8)
    axis.set_title("Era Correlation Distribution")
    axis.set_xlabel("CORR")
    axis.set_ylabel("Frequency")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_model_comparison(fold_metrics: pd.DataFrame, path: Path) -> Path:
    """Save a model comparison bar chart across folds."""
    aggregated = (
        fold_metrics.groupby("model_name", as_index=False)["mean_corr"]
        .mean()
        .sort_values("mean_corr", ascending=False)
    )
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(aggregated["model_name"], aggregated["mean_corr"])
    axis.set_title("Mean CORR by Model")
    axis.set_ylabel("Mean CORR")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_markdown_report(
    path: Path,
    *,
    run_name: str,
    summary: dict[str, Any],
    top_exposures: pd.Series,
    fold_metrics: pd.DataFrame,
) -> Path:
    """Write a compact markdown research report."""
    fold_table = [
        "| model_name | mean_corr | sharpe_like | max_drawdown | mean_abs_feature_exposure |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in fold_metrics.itertuples(index=False):
        fold_table.append(
            "| "
            f"{row.model_name} | {row.mean_corr:.6f} | {row.sharpe_like:.6f} | "
            f"{row.max_drawdown:.6f} | {row.mean_abs_feature_exposure:.6f} |"
        )

    exposure_table = [
        "| feature | abs_exposure |",
        "| --- | ---: |",
    ]
    for feature, value in top_exposures.items():
        exposure_table.append(f"| {feature} | {value:.6f} |")

    lines = [
        f"# Numerai Research Report: {run_name}",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.6f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Fold Results",
            "",
            *fold_table,
            "",
            "## Top Feature Exposures",
            "",
            *exposure_table,
            "",
            "## Notes",
            "",
            "- This report comes from walk-forward era validation, not random splitting.",
            "- Validation metrics are useful but still vulnerable to repeated tuning and leakage.",
            "- Neutralization here is a practical post-processing step, "
            "not a guarantee of higher live performance.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
