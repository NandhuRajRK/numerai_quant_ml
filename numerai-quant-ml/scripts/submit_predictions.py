"""Upload Numerai predictions with dry-run enabled by default."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from numerapi import NumerAPI

from numerai_quant.features import validate_prediction_frame
from numerai_quant.predict import read_predictions
from numerai_quant.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prediction_file",
        type=Path,
        help="CSV or parquet prediction file to upload.",
    )
    parser.add_argument("--submit", action="store_true", help="Actually upload predictions.")
    parser.add_argument("--env-file", default=".env", help="Path to dotenv file.")
    return parser.parse_args()


def required_env(name: str) -> str:
    """Read a required environment variable without logging its value."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    """Validate and optionally upload a prediction file."""
    setup_logging()
    args = parse_args()
    load_dotenv(args.env_file)

    frame = read_predictions(args.prediction_file)
    validate_prediction_frame(frame)

    public_id = required_env("NUMERAI_PUBLIC_ID")
    secret_key = required_env("NUMERAI_SECRET_KEY")
    model_id = required_env("NUMERAI_MODEL_ID")

    if not args.submit:
        LOGGER.info("Dry run successful. Prediction file is valid and credentials are present.")
        LOGGER.info("No upload was performed. Re-run with --submit to upload.")
        return

    api = NumerAPI(public_id=public_id, secret_key=secret_key)
    LOGGER.info("Uploading predictions for model id %s", model_id)
    api.upload_predictions(str(args.prediction_file), model_id=model_id)
    LOGGER.info("Upload complete.")


if __name__ == "__main__":
    main()
