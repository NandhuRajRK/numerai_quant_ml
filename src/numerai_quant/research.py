"""High-level research workflow for walk-forward backtests and final ensemble training."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from numerai_quant.artifacts import init_run_directory, save_dataframe, save_json, save_yaml
from numerai_quant.config import PipelineConfig
from numerai_quant.data import load_training_frame
from numerai_quant.ensemble import (
    blend_predictions,
    enabled_model_specs,
    fit_model,
    normalized_weights,
    predict_with_ensemble_bundle,
    raw_predict,
    save_ensemble_bundle,
)
from numerai_quant.features import (
    apply_feature_controls,
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
    build_walk_forward_folds,
    summarize_fold_result,
    validation_report,
)

LOGGER = logging.getLogger(__name__)


def _prediction_columns(frame: pd.DataFrame, prediction_col: str) -> list[str]:
    """Return prediction-like columns worth checkpointing."""
    selected = []
    for column in frame.columns:
        if column == prediction_col or column == "final_prediction":
            selected.append(column)
        elif column.endswith("_prediction"):
            selected.append(column)
    return selected


def _partial_leaderboard(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a running leaderboard from partial fold metrics."""
    if fold_metrics.empty:
        return pd.DataFrame()
    return (
        fold_metrics.groupby("model_name", as_index=False)[
            ["mean_corr", "sharpe_like", "max_drawdown", "mean_abs_feature_exposure"]
        ]
        .mean()
        .sort_values(["mean_corr", "sharpe_like"], ascending=[False, False])
    )


def _checkpoint_fold_frame(
    run_dir: Path,
    fold_number: int,
    fold_frame: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    """Persist the current fold's prediction progress without all raw features."""
    prediction_columns = _prediction_columns(fold_frame, config.prediction_col)
    checkpoint_frame = fold_frame[
        [config.id_col, config.era_col, config.target_col, *prediction_columns]
    ]
    save_dataframe(
        checkpoint_frame,
        run_dir / "checkpoints" / f"fold_{fold_number:02d}_progress.parquet",
    )


def _write_partial_artifacts(
    *,
    run_dir: Path,
    config: PipelineConfig,
    fold_rows: list[dict[str, float | int | str]],
    oof_prediction_frames: list[pd.DataFrame],
    current_fold_frame: pd.DataFrame | None = None,
    current_fold_number: int | None = None,
    status: dict[str, Any] | None = None,
) -> None:
    """Write partial metrics, predictions, and status for interrupted-run recovery."""
    if status is not None:
        save_json(status, run_dir / "status.json")

    if fold_rows:
        fold_metrics = pd.DataFrame(fold_rows)
        save_dataframe(fold_metrics, run_dir / "fold_metrics.partial.csv")
        leaderboard = _partial_leaderboard(fold_metrics)
        if not leaderboard.empty:
            save_dataframe(leaderboard, run_dir / "model_leaderboard.partial.csv")

    if oof_prediction_frames:
        save_dataframe(
            pd.concat(oof_prediction_frames, ignore_index=True),
            run_dir / "oof_predictions.partial.parquet",
        )

    if current_fold_frame is not None and current_fold_number is not None:
        _checkpoint_fold_frame(run_dir, current_fold_number, current_fold_frame, config)


def create_run_name(config: PipelineConfig, prefix: str = "research") -> str:
    """Build a stable run name."""
    return f"{prefix}_{config.dataset_version}_{timestamp_slug()}"


def _walk_forward_params(config: PipelineConfig) -> dict[str, Any]:
    return dict(config.raw["validation"]["walk_forward"])


def run_walk_forward_backtest(
    train_df: pd.DataFrame,
    config: PipelineConfig,
    *,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Run walk-forward validation across the configured model zoo."""
    train_df = ensure_identifier_column(train_df, id_col=config.id_col)
    feature_cols = detect_feature_columns(train_df)
    specs = enabled_model_specs(config)
    weights = normalized_weights(specs)
    walk_forward_params = _walk_forward_params(config)
    folds = build_walk_forward_folds(train_df, era_col=config.era_col, **walk_forward_params)
    run_dir = init_run_directory(
        config.path("artifacts_dir"),
        run_name or create_run_name(config, prefix="walkforward"),
    )
    save_yaml(config.raw, run_dir / "run_config.yaml")

    oof_prediction_frames: list[pd.DataFrame] = []
    final_report_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, float | int | str]] = []
    feature_selection_rows: list[dict[str, Any]] = []
    neutralization_cfg = dict(config.raw["advanced"]["neutralization"])
    use_neutralization = bool(neutralization_cfg["enabled"])
    feature_control_cfg = dict(config.raw.get("feature_selection", {}))
    total_folds = len(folds)
    _write_partial_artifacts(
        run_dir=run_dir,
        config=config,
        fold_rows=[],
        oof_prediction_frames=[],
        status={
            "stage": "initialized",
            "run_dir": str(run_dir),
            "total_folds": total_folds,
            "completed_folds": 0,
        },
    )

    for fold in folds:
        LOGGER.info(
            "Starting fold %s/%s | train eras=%s..%s (%s eras) | validation eras=%s..%s (%s eras)",
            fold.fold_number,
            total_folds,
            fold.train_eras[0],
            fold.train_eras[-1],
            len(fold.train_eras),
            fold.validation_eras[0],
            fold.validation_eras[-1],
            len(fold.validation_eras),
        )
        train_split = train_df[train_df[config.era_col].isin(fold.train_eras)].copy()
        valid_split = train_df[train_df[config.era_col].isin(fold.validation_eras)].copy()
        model_feature_cols, feature_summary = apply_feature_controls(
            train_split,
            feature_cols,
            target_col=config.target_col,
            config=feature_control_cfg,
        )
        feature_selection_rows.append(
            {
                "fold": fold.fold_number,
                **feature_summary,
            }
        )
        X_train = train_split[model_feature_cols]
        y_train = train_split[config.target_col]

        raw_valid_predictions: dict[str, Any] = {}
        fold_frame = valid_split[
            [config.id_col, config.era_col, config.target_col, *feature_cols]
        ].copy()

        for spec in specs:
            LOGGER.info(
                "Fold %s/%s | fitting %s with %s features",
                fold.fold_number,
                total_folds,
                spec["name"],
                len(model_feature_cols),
            )
            model = fit_model(spec, X_train, y_train)
            raw_valid_predictions[spec["name"]] = raw_predict(
                model,
                valid_split[model_feature_cols],
            )
            prediction_frame = make_prediction_frame(
                valid_split[config.id_col],
                raw_valid_predictions[spec["name"]],
                id_col=config.id_col,
                prediction_col=config.prediction_col,
            )
            candidate_frame = fold_frame.copy()
            candidate_frame[config.prediction_col] = prediction_frame[
                config.prediction_col
            ].to_numpy()
            report = validation_report(
                candidate_frame,
                feature_cols,
                era_col=config.era_col,
                target_col=config.target_col,
                prediction_col=config.prediction_col,
                exposure_sample_size=int(config.raw["validation"]["feature_exposure_sample_size"]),
            )
            fold_rows.append(summarize_fold_result(fold.fold_number, spec["name"], report))
            fold_frame[f"{spec['name']}_prediction"] = prediction_frame[
                config.prediction_col
            ].to_numpy()
            _write_partial_artifacts(
                run_dir=run_dir,
                config=config,
                fold_rows=fold_rows,
                oof_prediction_frames=oof_prediction_frames,
                current_fold_frame=fold_frame,
                current_fold_number=fold.fold_number,
                status={
                    "stage": "model_validated",
                    "current_fold": fold.fold_number,
                    "total_folds": total_folds,
                    "current_model": spec["name"],
                    "completed_folds": fold.fold_number - 1,
                },
            )

        ensemble_predictions = blend_predictions(raw_valid_predictions, weights)
        ensemble_frame = make_prediction_frame(
            valid_split[config.id_col],
            ensemble_predictions,
            id_col=config.id_col,
            prediction_col=config.prediction_col,
        )
        fold_frame["ensemble_prediction"] = ensemble_frame[config.prediction_col].to_numpy()

        ensemble_report_frame = fold_frame[
            [config.era_col, config.target_col, *feature_cols]
        ].copy()
        ensemble_report_frame[config.prediction_col] = fold_frame["ensemble_prediction"].to_numpy()
        ensemble_report = validation_report(
            ensemble_report_frame,
            feature_cols,
            era_col=config.era_col,
            target_col=config.target_col,
            prediction_col=config.prediction_col,
            exposure_sample_size=int(config.raw["validation"]["feature_exposure_sample_size"]),
        )
        fold_rows.append(summarize_fold_result(fold.fold_number, "ensemble", ensemble_report))

        final_prediction_column = "ensemble_prediction"
        final_label = "ensemble"
        if use_neutralization:
            fold_frame["ensemble_neutralized_prediction"] = apply_feature_neutralization(
                fold_frame,
                feature_cols,
                prediction_col="ensemble_prediction",
                sample_size=int(neutralization_cfg["top_n_features"]),
                proportion=float(neutralization_cfg["proportion"]),
            ).to_numpy()
            neutralized_report_frame = fold_frame[
                [config.era_col, config.target_col, *feature_cols]
            ].copy()
            neutralized_report_frame[config.prediction_col] = fold_frame[
                "ensemble_neutralized_prediction"
            ].to_numpy()
            neutralized_report = validation_report(
                neutralized_report_frame,
                feature_cols,
                era_col=config.era_col,
                target_col=config.target_col,
                prediction_col=config.prediction_col,
                exposure_sample_size=int(config.raw["validation"]["feature_exposure_sample_size"]),
            )
            fold_rows.append(
                summarize_fold_result(
                    fold.fold_number,
                    "ensemble_neutralized",
                    neutralized_report,
                )
            )
            final_prediction_column = "ensemble_neutralized_prediction"
            final_label = "ensemble_neutralized"

        fold_frame["final_prediction"] = fold_frame[final_prediction_column].to_numpy()
        final_report_frame = fold_frame[[config.era_col, config.target_col, *feature_cols]].copy()
        final_report_frame[config.prediction_col] = fold_frame["final_prediction"].to_numpy()
        final_report_frames.append(final_report_frame)

        oof_prediction_frame = fold_frame[
            [
                config.id_col,
                config.era_col,
                config.target_col,
                "ensemble_prediction",
                *(["ensemble_neutralized_prediction"] if use_neutralization else []),
                "final_prediction",
            ]
        ].copy()
        oof_prediction_frame["final_model_name"] = final_label
        oof_prediction_frame["fold"] = fold.fold_number
        oof_prediction_frames.append(oof_prediction_frame)
        _write_partial_artifacts(
            run_dir=run_dir,
            config=config,
            fold_rows=fold_rows,
            oof_prediction_frames=oof_prediction_frames,
            current_fold_frame=fold_frame,
            current_fold_number=fold.fold_number,
            status={
                "stage": "fold_completed",
                "current_fold": fold.fold_number,
                "total_folds": total_folds,
                "completed_folds": fold.fold_number,
                "final_model_name": final_label,
            },
        )
        LOGGER.info("Completed fold %s/%s", fold.fold_number, total_folds)

    oof_frame = pd.concat(oof_prediction_frames, ignore_index=True)
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
    leaderboard = _partial_leaderboard(fold_metrics)
    save_dataframe(oof_frame, run_dir / "oof_predictions.parquet")
    save_dataframe(fold_metrics, run_dir / "fold_metrics.csv")
    save_dataframe(leaderboard, run_dir / "model_leaderboard.csv")
    if feature_selection_rows:
        save_dataframe(
            pd.DataFrame(feature_selection_rows),
            run_dir / "feature_selection_summary.csv",
        )
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
    _write_partial_artifacts(
        run_dir=run_dir,
        config=config,
        fold_rows=fold_rows,
        oof_prediction_frames=oof_prediction_frames,
        status={
            "stage": "completed",
            "run_dir": str(run_dir),
            "total_folds": total_folds,
            "completed_folds": total_folds,
        },
    )

    return {
        "run_dir": run_dir,
        "feature_cols": feature_cols,
        "fold_metrics": fold_metrics,
        "leaderboard": leaderboard,
        "oof_frame": oof_frame,
        "final_report": final_report,
    }


def train_final_ensemble(
    config: PipelineConfig,
    *,
    artifact_name: str | None = None,
) -> dict[str, Any]:
    """Train the final ensemble for live prediction and snapshot it into a new run directory."""
    include_validation = bool(
        config.raw["advanced"]["train_on"]["include_validation_for_live_model"]
    )
    training_df = ensure_identifier_column(
        load_training_frame(config, include_validation=include_validation),
        id_col=config.id_col,
    )
    feature_cols = detect_feature_columns(training_df)
    model_feature_cols, feature_summary = apply_feature_controls(
        training_df,
        feature_cols,
        target_col=config.target_col,
        config=dict(config.raw.get("feature_selection", {})),
    )
    specs = enabled_model_specs(config)
    models = {
        spec["name"]: fit_model(
            spec,
            training_df[model_feature_cols],
            training_df[config.target_col],
        )
        for spec in specs
    }

    bundle_dir = save_ensemble_bundle(
        models,
        model_feature_cols,
        config,
        artifact_name=artifact_name,
    )
    run_dir = init_run_directory(
        config.path("artifacts_dir"),
        create_run_name(config, prefix="finalfit"),
    )
    shutil.copytree(bundle_dir, run_dir / "models" / bundle_dir.name, dirs_exist_ok=True)
    save_yaml(config.raw, run_dir / "run_config.yaml")
    save_json(feature_summary, run_dir / "feature_selection_summary.json")
    return {
        "bundle_dir": bundle_dir,
        "run_dir": run_dir,
        "feature_cols": model_feature_cols,
    }


def build_live_predictions_from_bundle(
    bundle_dir: Path,
    live_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load an ensemble bundle and return Numerai-formatted live predictions."""
    from numerai_quant.ensemble import load_ensemble_bundle
    from numerai_quant.validation import apply_feature_neutralization

    models, metadata = load_ensemble_bundle(bundle_dir)
    predictions, raw_by_model = predict_with_ensemble_bundle(models, metadata, live_df)
    neutralization_cfg = dict(metadata.get("neutralization", {}))

    if neutralization_cfg.get("enabled", False):
        feature_cols = list(metadata["feature_columns"])
        predictions[str(metadata["prediction_col"])] = apply_feature_neutralization(
            live_df.assign(
                **{
                    str(metadata["prediction_col"]): predictions[
                        str(metadata["prediction_col"])
                    ]
                }
            ),
            feature_cols,
            prediction_col=str(metadata["prediction_col"]),
            sample_size=int(neutralization_cfg["top_n_features"]),
            proportion=float(neutralization_cfg["proportion"]),
        ).to_numpy()
    return predictions, {"metadata": metadata, "raw_by_model": raw_by_model}
