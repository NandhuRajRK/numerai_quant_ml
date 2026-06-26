"""Train and save the LightGBM baseline model."""

from __future__ import annotations

import argparse
import logging

from numerai_quant.config import load_config
from numerai_quant.data import load_parquet, raw_dataset_path
from numerai_quant.modeling import save_model, train_lightgbm_baseline
from numerai_quant.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    """Train the baseline model."""
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    train_path = raw_dataset_path(config, config.numerai_file("train_file"))
    train_df = load_parquet(train_path)
    model, feature_cols = train_lightgbm_baseline(train_df, config)
    save_model(model, feature_cols, config)
    LOGGER.info("Training complete.")


if __name__ == "__main__":
    main()
