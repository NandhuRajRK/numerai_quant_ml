"""Run the full portfolio pipeline: walk-forward backtest, final fit, and live predictions."""

from __future__ import annotations

import argparse
import logging

from numerai_quant.config import load_config
from numerai_quant.data import download_live_data, load_parquet, load_training_frame
from numerai_quant.predict import save_predictions
from numerai_quant.research import (
    build_live_predictions_from_bundle,
    run_walk_forward_backtest,
    train_final_ensemble,
)
from numerai_quant.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use the existing live parquet file instead of downloading a fresh copy.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full advanced Numerai workflow."""
    setup_logging()
    args = parse_args()
    config = load_config(args.config)

    train_df = load_training_frame(config, include_validation=False)
    backtest = run_walk_forward_backtest(train_df, config)
    final_fit = train_final_ensemble(config)

    live_path = (
        config.path("raw_data_dir") / config.dataset_version / config.numerai_file("live_file")
        if args.skip_download
        else download_live_data(config, overwrite=True)
    )
    live_df = load_parquet(live_path)
    predictions, _ = build_live_predictions_from_bundle(final_fit["bundle_dir"], live_df)
    output_path = config.path("predictions_dir") / "live_predictions_portfolio.csv"
    save_predictions(predictions, output_path)

    LOGGER.info("Walk-forward artifacts: %s", backtest["run_dir"])
    LOGGER.info("Final ensemble bundle: %s", final_fit["bundle_dir"])
    LOGGER.info("Live predictions: %s", output_path)


if __name__ == "__main__":
    main()
