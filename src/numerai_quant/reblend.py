"""Recompute ensemble artifacts from cached fold predictions without retraining."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from numerai_quant.artifacts import init_run_directory, save_dataframe, save_json, save_yaml
from numerai_quant.config import PipelineConfig, load_config_mapping
from numerai_quant.data import load_training_frame
from numerai_quant.ensemble import enabled_model_specs, normalized_weights
from numerai_quant.features import (
    detect_feature_columns,
    ensure_identifier_column,
    make_prediction_frame,
)
from numerai_quant.reporting import (
    plot_correlation_histogram,
    plot_cumulative_correlation,
    plot_model_comparison,
    write_markdown_report,
)
from numerai_quant.utils import timestamp_slug
from numerai_quant.validation import (
    apply_feature_neutralization,
    summarize_fold_result,
    validation_report,
)

LOGGER = logging.getLogger(__name__)


def _load_run_config(run_dir: Path, root_dir: Path) -> PipelineConfig:
    with (run_dir / "run_config.yaml").open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid run config in {run_dir / 'run_config.yaml'}")
    return load_config_mapping(raw, root_dir=root_dir)


def _checkpoint_fold_number(path: Path) -> int:
    match = re.search(r"fold_(\d+)_progress", path.stem)
    if match is None:
        raise ValueError(f"Could not parse fold number from {path}")
    return int(match.group(1))


def _checkpoint_paths(run_dir: Path) -> list[Path]:
    checkpoints = sorted(
        (run_dir / "checkpoints").glob("fold_*_progress.parquet"),
        key=_checkpoint_fold_number,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No fold checkpoints found in {run_dir / 'checkpoints'}")
    return checkpoints


def _select_model_prediction_columns(
    frame: pd.DataFrame,
    model_names: list[str],
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for name in model_names:
        raw_col = f"{name}_raw_prediction"
        normalized_col = f"{name}_prediction"
        if raw_col in frame.columns:
            selected[name] = raw_col
        elif normalized_col in frame.columns:
            selected[name] = normalized_col
        else:
            raise ValueError(
                f"Missing cached predictions for model {name!r}. "
                f"Expected one of {raw_col!r} or {normalized_col!r}."
            )
    return selected


def _source_model_metrics(run_dir: Path, model_names: list[str]) -> pd.DataFrame:
    fold_metrics = pd.read_csv(run_dir / "fold_metrics.csv")
    return fold_metrics[fold_metrics["model_name"].isin(model_names)].copy()


def _validation_feature_frame(
    source_config: PipelineConfig,
    *,
    ids: pd.Series,
) -> tuple[pd.DataFrame, list[str]]:
    training_df = ensure_identifier_column(
        load_training_frame(source_config, include_validation=False),
        id_col=source_config.id_col,
    )
    feature_cols = detect_feature_columns(training_df)
    filtered = training_df[training_df[source_config.id_col].isin(ids)].copy()
    return filtered, feature_cols


def _reblend_run_name(source_run_dir: Path) -> str:
    return f"reblend_{source_run_dir.name}_{timestamp_slug()}"


def recompute_ensemble_from_cached_predictions(
    *,
    source_run_dir: Path,
    config: PipelineConfig,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Recompute ensemble artifacts from cached fold predictions."""
    source_config = _load_run_config(source_run_dir, config.root_dir)
    specs = enabled_model_specs(config)
    model_names = [str(spec["name"]) for spec in specs]
    weights = normalized_weights(specs)
    checkpoints = _checkpoint_paths(source_run_dir)
    source_model_rows = _source_model_metrics(source_run_dir, model_names)

    checkpoint_frames = [pd.read_parquet(path) for path in checkpoints]
    oof_cached = pd.concat(checkpoint_frames, ignore_index=True)
    feature_lookup, feature_cols = _validation_feature_frame(
        source_config,
        ids=oof_cached[config.id_col],
    )

    run_dir = init_run_directory(
        config.path("artifacts_dir"),
        run_name or _reblend_run_name(source_run_dir),
    )
    save_yaml(config.raw, run_dir / "run_config.yaml")
    save_json(
        {
            "source_run_dir": str(source_run_dir),
            "reblend_model_names": model_names,
            "weights": weights,
        },
        run_dir / "reblend_source.json",
    )

    fold_rows = source_model_rows.to_dict(orient="records")
    oof_frames: list[pd.DataFrame] = []
    final_report_frames: list[pd.DataFrame] = []
    neutralization_cfg = dict(config.raw["advanced"]["neutralization"])
    use_neutralization = bool(neutralization_cfg["enabled"])

    for checkpoint_path, checkpoint_frame in zip(checkpoints, checkpoint_frames, strict=True):
        fold_number = _checkpoint_fold_number(checkpoint_path)
        prediction_columns = _select_model_prediction_columns(checkpoint_frame, model_names)
        blended_values = sum(
            weights[name] * checkpoint_frame[column].to_numpy()
            for name, column in prediction_columns.items()
        )
        ensemble_frame = make_prediction_frame(
            checkpoint_frame[config.id_col],
            blended_values,
            id_col=config.id_col,
            prediction_col=config.prediction_col,
        )

        report_frame = checkpoint_frame[
            [config.id_col, config.era_col, config.target_col]
        ].merge(
            feature_lookup[[config.id_col, *feature_cols]],
            on=config.id_col,
            how="left",
        )
        report_frame[config.prediction_col] = ensemble_frame[config.prediction_col].to_numpy()
        ensemble_report = validation_report(
            report_frame,
            feature_cols,
            era_col=config.era_col,
            target_col=config.target_col,
            prediction_col=config.prediction_col,
            exposure_sample_size=int(config.raw["validation"]["feature_exposure_sample_size"]),
        )
        fold_rows.append(summarize_fold_result(fold_number, "ensemble", ensemble_report))

        final_prediction = ensemble_frame[config.prediction_col].to_numpy()
        final_model_name = "ensemble"
        oof_frame = checkpoint_frame[[config.id_col, config.era_col, config.target_col]].copy()
        for name, column in prediction_columns.items():
            oof_frame[f"{name}_prediction"] = checkpoint_frame[column].to_numpy()

        if use_neutralization:
            neutralized_values = apply_feature_neutralization(
                report_frame,
                feature_cols,
                prediction_col=config.prediction_col,
                sample_size=int(neutralization_cfg["top_n_features"]),
                proportion=float(neutralization_cfg["proportion"]),
            ).to_numpy()
            neutralized_report_frame = report_frame.copy()
            neutralized_report_frame[config.prediction_col] = neutralized_values
            neutralized_report = validation_report(
                neutralized_report_frame,
                feature_cols,
                era_col=config.era_col,
                target_col=config.target_col,
                prediction_col=config.prediction_col,
                exposure_sample_size=int(config.raw["validation"]["feature_exposure_sample_size"]),
            )
            fold_rows.append(
                summarize_fold_result(fold_number, "ensemble_neutralized", neutralized_report)
            )
            oof_frame["ensemble_neutralized_prediction"] = neutralized_values
            final_prediction = neutralized_values
            final_model_name = "ensemble_neutralized"

        oof_frame["ensemble_prediction"] = ensemble_frame[config.prediction_col].to_numpy()
        oof_frame["final_prediction"] = final_prediction
        oof_frame["final_model_name"] = final_model_name
        oof_frame["fold"] = fold_number
        oof_frames.append(oof_frame)

        final_frame = report_frame.copy()
        final_frame[config.prediction_col] = final_prediction
        final_report_frames.append(final_frame)

    oof_result = pd.concat(oof_frames, ignore_index=True)
    final_report_frame = pd.concat(final_report_frames, ignore_index=True)
    final_report = validation_report(
        final_report_frame,
        feature_cols,
        era_col=config.era_col,
        target_col=config.target_col,
        prediction_col=config.prediction_col,
        exposure_sample_size=int(config.raw["validation"]["feature_exposure_sample_size"]),
    )

    fold_metrics = pd.DataFrame(fold_rows)
    leaderboard = (
        fold_metrics.groupby("model_name", as_index=False)[
            ["mean_corr", "sharpe_like", "max_drawdown", "mean_abs_feature_exposure"]
        ]
        .mean()
        .sort_values(["mean_corr", "sharpe_like"], ascending=[False, False])
    )

    save_dataframe(oof_result, run_dir / "oof_predictions.parquet")
    save_dataframe(fold_metrics, run_dir / "fold_metrics.csv")
    save_dataframe(leaderboard, run_dir / "model_leaderboard.csv")
    feature_exposure_frame = (
        final_report["feature_exposure"]
        .rename("abs_exposure")
        .rename_axis("feature")
        .reset_index()
    )
    save_dataframe(feature_exposure_frame, run_dir / "feature_exposure.csv")
    save_json(final_report["summary"], run_dir / "summary.json")
    plot_cumulative_correlation(
        final_report["era_correlations"],
        run_dir / "plots" / "cumulative_corr.png",
    )
    plot_correlation_histogram(
        final_report["era_correlations"],
        run_dir / "plots" / "era_corr_histogram.png",
    )
    plot_model_comparison(fold_metrics, run_dir / "plots" / "model_comparison.png")
    write_markdown_report(
        run_dir / "report.md",
        run_name=run_dir.name,
        summary=final_report["summary"],
        top_exposures=final_report["feature_exposure"].head(
            int(config.raw["reporting"]["top_feature_exposure_count"])
        ),
        fold_metrics=leaderboard,
    )
    save_json(
        {
            "stage": "completed",
            "source_run_dir": str(source_run_dir),
            "total_folds": len(checkpoints),
            "completed_folds": len(checkpoints),
        },
        run_dir / "status.json",
    )

    return {
        "run_dir": run_dir,
        "leaderboard": leaderboard,
        "fold_metrics": fold_metrics,
        "oof_frame": oof_result,
        "final_report": final_report,
    }
