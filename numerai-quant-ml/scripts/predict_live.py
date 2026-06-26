"""Download live data and generate Numerai submission predictions."""

from __future__ import annotations

import argparse
import logging

from numerapi import NumerAPI

from numerai_quant.config import load_config
from numerai_quant.data import download_live_data, load_parquet
from numerai_quant.ensemble import ensemble_bundle_dir
from numerai_quant.modeling import load_model, predict_with_model
from numerai_quant.predict import save_predictions
from numerai_quant.research import build_live_predictions_from_bundle
from numerai_quant.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use existing live parquet file.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format.",
    )
    parser.add_argument(
        "--use-baseline",
        action="store_true",
        help="Use the original single-model baseline instead of the saved ensemble bundle.",
    )
    parser.add_argument(
        "--artifact-name",
        default=None,
        help="Override the advanced ensemble bundle directory name.",
    )
    return parser.parse_args()


def current_round() -> int | str:
    """Return current Numerai round if numerapi exposes it, otherwise ``unknown``."""
    try:
        return NumerAPI().get_current_round()
    except Exception as exc:  # pragma: no cover - network/API defensive fallback
        LOGGER.warning("Could not fetch current round from numerapi: %s", exc)
        return "unknown"


def main() -> None:
    """Generate live predictions with the saved baseline model."""
    setup_logging()
    args = parse_args()
    config = load_config(args.config)

    live_path = (
        config.path("raw_data_dir") / config.dataset_version / config.numerai_file("live_file")
        if args.no_download
        else download_live_data(config, overwrite=True)
    )
    live_df = load_parquet(live_path)
    if args.use_baseline:
        model, feature_cols = load_model(config)
        predictions = predict_with_model(model, live_df, feature_cols, config)
    else:
        bundle_dir = ensemble_bundle_dir(config, artifact_name=args.artifact_name)
        predictions, _ = build_live_predictions_from_bundle(bundle_dir, live_df)

    round_number = current_round()
    suffix = args.format
    output_path = config.path("predictions_dir") / f"live_predictions_{round_number}.{suffix}"
    save_predictions(predictions, output_path)
    LOGGER.info("Saved live predictions to %s", output_path)


if __name__ == "__main__":
    main()
