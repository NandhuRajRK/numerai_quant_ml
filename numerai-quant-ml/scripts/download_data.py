"""Download Numerai train, validation, and live data."""

from __future__ import annotations

import argparse
import logging

from numerai_quant.config import load_config
from numerai_quant.data import download_training_validation_live
from numerai_quant.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files.")
    return parser.parse_args()


def main() -> None:
    """Download configured Numerai datasets."""
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    paths = download_training_validation_live(config, overwrite=args.overwrite)
    for path in paths:
        LOGGER.info("Ready: %s", path)


if __name__ == "__main__":
    main()
