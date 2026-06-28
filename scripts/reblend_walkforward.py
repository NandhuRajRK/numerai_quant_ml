"""Recompute ensemble artifacts from cached fold predictions without retraining."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from numerai_quant.config import load_config
from numerai_quant.reblend import recompute_ensemble_from_cached_predictions
from numerai_quant.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to new YAML config.",
    )
    parser.add_argument(
        "--source-run",
        required=True,
        help="Path to an existing walk-forward artifact directory with checkpoints.",
    )
    parser.add_argument("--run-name", default=None, help="Optional output run directory name.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    results = recompute_ensemble_from_cached_predictions(
        source_run_dir=Path(args.source_run).resolve(),
        config=config,
        run_name=args.run_name,
    )
    LOGGER.info("Saved reblended artifacts to %s", results["run_dir"])
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
