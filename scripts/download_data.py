"""Download Numerai train, validation, and live data."""

from __future__ import annotations

import argparse
import logging

from numerai_quant.config import load_config
from numerai_quant.data import execute_download_plan, resolve_download_plan
from numerai_quant.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--train-only",
        action="store_true",
        help="Download only the train parquet. This is the default behavior.",
    )
    selection.add_argument(
        "--validation-only",
        action="store_true",
        help="Download only the validation parquet.",
    )
    selection.add_argument(
        "--live-only",
        action="store_true",
        help="Download only the live parquet.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Download train first, then validation and live.",
    )
    parser.add_argument(
        "--parallel-secondary",
        action="store_true",
        help="When used with --all, download validation and live in parallel after train.",
    )
    return parser.parse_args()


def main() -> None:
    """Download configured Numerai datasets."""
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    plan = resolve_download_plan(args)
    LOGGER.info(
        "Download plan: primary=%s secondary=%s parallel_secondary=%s",
        plan.primary_file_keys,
        plan.secondary_file_keys,
        plan.parallel_secondary,
    )
    paths = execute_download_plan(config, plan, overwrite=args.overwrite)
    for path in paths:
        LOGGER.info("Ready: %s", path)


if __name__ == "__main__":
    main()
