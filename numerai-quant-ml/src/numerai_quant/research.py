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
from numerai_quant.features import detect_feature_columns, make_prediction_frame
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
    feature_cols = detect_feature_columns(train_df)
    specs = enabled_model_specs(config)
    weights = normalized_weights(specs)
    walk_forward_params = _walk_forward_params(config)
    folds = build_walk_forward_folds(train_df, era_col=config.era_col, **walk_forward_params)
    run_dir = init_run_directory(
        config.path("artifacts_dir"),
        run_name or create_run_name(config, prefix="walkforward"),
    )

    oof_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, float | int | str]] = []
    neutralization_cfg = dict(config.raw["advanced"]["neutralization"])
    use_neutralization = bool(neutralization_cfg["enabled"])

    for fold in folds:
        train_split = train_df[train_df[config.era_col].isin(fold.train_eras)].copy()
        valid_split = train_df[train_df[config.era_col].isin(fold.validation_eras)].copy()
        X_train = train_split[feature_cols]
        y_train = train_split[config.target_col]

        raw_valid_predictions: dict[str, Any] = {}
        fold_frame = valid_split[
            [config.id_col, config.era_col, config.target_col, *feature_cols]
        ].copy()

        for spec in specs:
            model = fit_model(spec, X_train, y_train)
            raw_valid_predictions[spec["name"]] = raw_predict(model, valid_split[feature_cols])
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
        fold_frame["final_model_name"] = final_label
        fold_frame["fold"] = fold.fold_number
        oof_frames.append(fold_frame)

    oof_frame = pd.concat(oof_frames, ignore_index=True)
    final_report_frame = oof_frame[[config.era_col, config.target_col, *feature_cols]].copy()
    final_report_frame[config.prediction_col] = oof_frame["final_prediction"].to_numpy()
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

    save_yaml(config.raw, run_dir / "run_config.yaml")
    save_dataframe(oof_frame, run_dir / "oof_predictions.parquet")
    save_dataframe(fold_metrics, run_dir / "fold_metrics.csv")
    save_dataframe(leaderboard, run_dir / "model_leaderboard.csv")
    save_dataframe(
        final_report["feature_exposure"].rename("abs_exposure").reset_index(names="feature"),
        run_dir / "feature_exposure.csv",
    )
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
    training_df = load_training_frame(config, include_validation=include_validation)
    feature_cols = detect_feature_columns(training_df)
    specs = enabled_model_specs(config)
    models = {
        spec["name"]: fit_model(spec, training_df[feature_cols], training_df[config.target_col])
        for spec in specs
    }

    bundle_dir = save_ensemble_bundle(models, feature_cols, config, artifact_name=artifact_name)
    run_dir = init_run_directory(
        config.path("artifacts_dir"),
        create_run_name(config, prefix="finalfit"),
    )
    shutil.copytree(bundle_dir, run_dir / "models" / bundle_dir.name, dirs_exist_ok=True)
    save_yaml(config.raw, run_dir / "run_config.yaml")
    return {
        "bundle_dir": bundle_dir,
        "run_dir": run_dir,
        "feature_cols": feature_cols,
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
