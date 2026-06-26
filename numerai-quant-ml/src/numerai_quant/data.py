"""Numerai dataset download and loading helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from numerapi import NumerAPI

from numerai_quant.config import PipelineConfig
from numerai_quant.utils import ensure_directories

LOGGER = logging.getLogger(__name__)


def dataset_name(config: PipelineConfig, file_name: str) -> str:
    """Build a Numerai dataset identifier such as ``v5.2/train.parquet``."""
    return f"{config.dataset_version}/{file_name}"


def raw_dataset_path(config: PipelineConfig, file_name: str) -> Path:
    """Return the local path for a raw Numerai dataset file."""
    return config.path("raw_data_dir") / config.dataset_version / file_name


def download_dataset(
    config: PipelineConfig,
    file_key: str,
    *,
    overwrite: bool = False,
    napi: NumerAPI | None = None,
) -> Path:
    """Download a configured Numerai dataset file with numerapi."""
    file_name = config.numerai_file(file_key)
    destination = raw_dataset_path(config, file_name)
    ensure_directories([destination.parent])

    if destination.exists() and not overwrite:
        LOGGER.info("Dataset already exists, skipping download: %s", destination)
        return destination

    api = napi or NumerAPI()
    remote_name = dataset_name(config, file_name)
    LOGGER.info("Downloading Numerai dataset %s to %s", remote_name, destination)
    api.download_dataset(remote_name, str(destination))
    return destination


def download_training_validation_live(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Download train, validation, and live datasets."""
    return [
        download_dataset(config, "train_file", overwrite=overwrite),
        download_dataset(config, "validation_file", overwrite=overwrite),
        download_dataset(config, "live_file", overwrite=overwrite),
    ]


def download_live_data(config: PipelineConfig, *, overwrite: bool = True) -> Path:
    """Download the configured live dataset."""
    return download_dataset(config, "live_file", overwrite=overwrite)


def load_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a parquet file into a DataFrame."""
    LOGGER.info("Loading parquet: %s", path)
    return pd.read_parquet(path, columns=columns)


def load_training_frame(
    config: PipelineConfig,
    *,
    include_validation: bool = False,
) -> pd.DataFrame:
    """Load training data, optionally appending the validation rows with targets."""
    train_path = raw_dataset_path(config, config.numerai_file("train_file"))
    train_df = load_parquet(train_path)

    if not include_validation:
        return train_df

    validation_path = raw_dataset_path(config, config.numerai_file("validation_file"))
    validation_df = load_parquet(validation_path)
    if config.target_col not in validation_df.columns:
        LOGGER.warning(
            "Validation file lacks target column. "
            "Using train data only for final fitting."
        )
        return train_df

    LOGGER.info("Appending validation data to training frame for final fitting.")
    return pd.concat([train_df, validation_df], ignore_index=True)
