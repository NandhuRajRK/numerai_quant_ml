"""Generate validation predictions and report era-wise diagnostics."""

from __future__ import annotations

import argparse
import logging

from numerai_quant.config import load_config
from numerai_quant.data import load_parquet, raw_dataset_path
from numerai_quant.features import detect_feature_columns, ensure_identifier_column
from numerai_quant.modeling import load_model, predict_with_model
from numerai_quant.predict import save_predictions
from numerai_quant.utils import require_columns, setup_logging
from numerai_quant.validation import validation_report

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    """Validate the saved baseline model by era."""
    setup_logging()
    args = parse_args()
    config = load_config(args.config)

    model, feature_cols = load_model(config)
    validation_path = raw_dataset_path(config, config.numerai_file("validation_file"))
    validation_df = ensure_identifier_column(
        load_parquet(validation_path),
        id_col=config.id_col,
    )
    require_columns(
        set(validation_df.columns),
        {config.id_col, config.era_col, config.target_col},
        "Validation data",
    )

    prediction_frame = predict_with_model(model, validation_df, feature_cols, config)
    output_path = config.path("predictions_dir") / "validation_predictions.csv"
    save_predictions(prediction_frame, output_path)
    LOGGER.info("Saved validation predictions to %s", output_path)

    validation_features = detect_feature_columns(validation_df)
    report_frame = validation_df[[config.era_col, config.target_col, *validation_features]].copy()
    report_frame[config.prediction_col] = prediction_frame[config.prediction_col].to_numpy()
    report = validation_report(
        report_frame,
        feature_cols,
        era_col=config.era_col,
        target_col=config.target_col,
        prediction_col=config.prediction_col,
        exposure_sample_size=int(config.raw["validation"]["feature_exposure_sample_size"]),
    )

    LOGGER.info("Validation summary:")
    for key, value in report["summary"].items():
        LOGGER.info("  %s: %.6f", key, value)

    era_corr_path = config.path("predictions_dir") / "validation_era_correlations.csv"
    report["era_correlations"].rename("corr").to_csv(era_corr_path, header=True)
    LOGGER.info("Saved era correlations to %s", era_corr_path)


if __name__ == "__main__":
    main()
