"""Numerai dataset download and loading helpers."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from numerapi import NumerAPI

from numerai_quant.config import PipelineConfig
from numerai_quant.utils import ensure_directories

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadPlan:
    """Download plan for one or more Numerai dataset files."""

    primary_file_keys: list[str]
    secondary_file_keys: list[str]
    parallel_secondary: bool = False


def resolve_download_plan(options: Any) -> DownloadPlan:
    """Resolve CLI-like options into a dependency-aware download plan."""
    if getattr(options, "all", False):
        return DownloadPlan(
            primary_file_keys=["train_file"],
            secondary_file_keys=["validation_file", "live_file"],
            parallel_secondary=bool(getattr(options, "parallel_secondary", False)),
        )
    if getattr(options, "validation_only", False):
        return DownloadPlan(primary_file_keys=["validation_file"], secondary_file_keys=[])
    if getattr(options, "live_only", False):
        return DownloadPlan(primary_file_keys=["live_file"], secondary_file_keys=[])
    return DownloadPlan(primary_file_keys=["train_file"], secondary_file_keys=[])


def dataset_name(config: PipelineConfig, file_name: str) -> str:
    """Build a Numerai dataset identifier such as ``v5.2/train.parquet``."""
    return f"{config.dataset_version}/{file_name}"


def raw_dataset_path(config: PipelineConfig, file_name: str) -> Path:
    """Return the local path for a raw Numerai dataset file."""
    return config.path("raw_data_dir") / config.dataset_version / file_name


def temp_dataset_path(config: PipelineConfig, file_name: str) -> Path:
    """Return the temporary download path used by numerapi."""
    return raw_dataset_path(config, file_name).with_name(f"{file_name}.temp")


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
    temp_path = temp_dataset_path(config, file_name)
    ensure_directories([destination.parent])

    if destination.exists() and not overwrite:
        LOGGER.info("Dataset already exists, skipping download: %s", destination)
        return destination

    if overwrite:
        if destination.exists():
            LOGGER.info("Removing existing completed dataset before re-download: %s", destination)
            destination.unlink()
        if temp_path.exists():
            LOGGER.info("Removing existing partial download before re-download: %s", temp_path)
            temp_path.unlink()

    if temp_path.exists():
        partial_size_gb = temp_path.stat().st_size / (1024**3)
        LOGGER.info(
            "Found partial download for %s at %.2f GiB. numerapi will attempt to resume it.",
            file_name,
            partial_size_gb,
        )

    api = napi or NumerAPI()
    remote_name = dataset_name(config, file_name)
    LOGGER.info("Starting Numerai dataset download: %s -> %s", remote_name, destination)
    start = perf_counter()
    api.download_dataset(remote_name, str(destination))
    elapsed_seconds = perf_counter() - start
    final_size_gb = destination.stat().st_size / (1024**3)
    LOGGER.info(
        "Finished %s in %.1f minutes (%.2f GiB).",
        file_name,
        elapsed_seconds / 60.0,
        final_size_gb,
    )
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


def execute_download_plan(
    config: PipelineConfig,
    plan: DownloadPlan,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Execute a download plan with optional parallel secondary downloads."""
    api = NumerAPI()
    completed: list[Path] = []

    for file_key in plan.primary_file_keys:
        completed.append(download_dataset(config, file_key, overwrite=overwrite, napi=api))

    if not plan.secondary_file_keys:
        return completed

    if plan.parallel_secondary:
        LOGGER.info("Downloading secondary datasets in parallel: %s", plan.secondary_file_keys)
        with ThreadPoolExecutor(max_workers=len(plan.secondary_file_keys)) as executor:
            futures = [
                executor.submit(download_dataset, config, file_key, overwrite=overwrite)
                for file_key in plan.secondary_file_keys
            ]
            for future in futures:
                completed.append(future.result())
        return completed

    LOGGER.info("Downloading secondary datasets sequentially: %s", plan.secondary_file_keys)
    for file_key in plan.secondary_file_keys:
        completed.append(download_dataset(config, file_key, overwrite=overwrite, napi=api))
    return completed


def download_live_data(config: PipelineConfig, *, overwrite: bool = True) -> Path:
    """Download the configured live dataset."""
    return download_dataset(config, "live_file", overwrite=overwrite)


def load_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a parquet file into a DataFrame."""
    LOGGER.info("Loading parquet: %s", path)
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def load_parquet_with_era_limit(
    path: Path,
    *,
    era_col: str,
    era_limit: int,
) -> pd.DataFrame:
    """Load only the first few eras from a parquet file using pyarrow filters."""
    LOGGER.info("Loading first %s eras from parquet: %s", era_limit, path)
    era_frame = pd.read_parquet(path, columns=[era_col], engine="pyarrow")
    selected_eras = era_frame[era_col].drop_duplicates().tolist()[:era_limit]
    if not selected_eras:
        raise ValueError(f"No eras found in parquet file: {path}")
    return pd.read_parquet(
        path,
        filters=[(era_col, "in", selected_eras)],
        engine="pyarrow",
    )


def load_training_frame(
    config: PipelineConfig,
    *,
    include_validation: bool = False,
) -> pd.DataFrame:
    """Load training data, optionally appending the validation rows with targets."""
    train_path = raw_dataset_path(config, config.numerai_file("train_file"))
    runtime = dict(config.raw.get("runtime", {}))
    train_era_limit = runtime.get("train_era_limit")
    if train_era_limit:
        train_df = load_parquet_with_era_limit(
            train_path,
            era_col=config.era_col,
            era_limit=int(train_era_limit),
        )
    else:
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
