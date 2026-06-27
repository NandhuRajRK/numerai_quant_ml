"""Train the final portfolio ensemble for live Numerai predictions."""

from __future__ import annotations

import argparse
import logging

from numerai_quant.config import load_config
from numerai_quant.research import train_final_ensemble
from numerai_quant.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--artifact-name",
        default=None,
        help="Override the stable ensemble bundle directory name.",
    )
    return parser.parse_args()


def main() -> None:
    """Train and save the production-style ensemble bundle."""
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    results = train_final_ensemble(config, artifact_name=args.artifact_name)
    LOGGER.info("Saved live ensemble bundle to %s", results["bundle_dir"])
    LOGGER.info("Snapshot artifacts stored at %s", results["run_dir"])


if __name__ == "__main__":
    main()
