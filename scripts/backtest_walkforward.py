"""Run walk-forward era validation across the configured model zoo."""

from __future__ import annotations

import argparse
import logging

from numerai_quant.config import load_config
from numerai_quant.data import load_training_frame
from numerai_quant.research import run_walk_forward_backtest
from numerai_quant.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    """Execute the walk-forward research backtest."""
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    train_df = load_training_frame(config, include_validation=False)
    results = run_walk_forward_backtest(train_df, config)
    LOGGER.info("Saved walk-forward artifacts to %s", results["run_dir"])
    LOGGER.info("Top models:")
    for row in results["leaderboard"].head(5).itertuples(index=False):
        LOGGER.info(
            "  %s | mean_corr=%.6f | sharpe_like=%.6f",
            row.model_name,
            row.mean_corr,
            row.sharpe_like,
        )


if __name__ == "__main__":
    main()
