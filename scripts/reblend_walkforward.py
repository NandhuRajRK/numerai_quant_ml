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
    parser.add_argument(
        "--optimize-weights",
        action="store_true",
        help="Search for better ensemble weights from cached fold predictions.",
    )
    parser.add_argument(
        "--objective",
        default="mean_corr",
        choices=["mean_corr", "sharpe_like"],
        help="Optimization objective when --optimize-weights is enabled.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.05,
        help="Simplex grid step for small model sets, for example 0.1 or 0.05.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=2000,
        help="Random Dirichlet trials for larger model sets.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    results = recompute_ensemble_from_cached_predictions(
        source_run_dir=Path(args.source_run).resolve(),
        config=config,
        run_name=args.run_name,
        optimize_weights=args.optimize_weights,
        objective=args.objective,
        grid_step=args.grid_step,
        random_trials=args.random_trials,
    )
    LOGGER.info("Saved reblended artifacts to %s", results["run_dir"])
    LOGGER.info("Blend weights: %s", results["weights"])
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
