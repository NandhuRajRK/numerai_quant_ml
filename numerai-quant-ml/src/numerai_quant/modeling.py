"""Model training and persistence for the LightGBM baseline."""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml

from numerai_quant.config import PipelineConfig
from numerai_quant.features import detect_feature_columns, make_prediction_frame
from numerai_quant.utils import ensure_directories, require_columns

LOGGER = logging.getLogger(__name__)


def train_lightgbm_baseline(
    train_df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[lgb.LGBMRegressor, list[str]]:
    """Train a simple LightGBM regression baseline on Numerai features."""
    feature_cols = detect_feature_columns(train_df)
    require_columns(set(train_df.columns), {config.target_col}, "Training data")

    model = lgb.LGBMRegressor(**config.model_params)
    LOGGER.info(
        "Training LightGBM baseline on %s rows and %s features",
        len(train_df),
        len(feature_cols),
    )
    model.fit(train_df[feature_cols], train_df[config.target_col])
    return model, feature_cols


def model_artifact_path(config: PipelineConfig) -> Path:
    """Return the path to the saved LightGBM model."""
    return config.path("models_dir") / f"{config.model_name}.txt"


def metadata_artifact_path(config: PipelineConfig) -> Path:
    """Return the path to the saved model metadata."""
    return config.path("models_dir") / f"{config.model_name}_metadata.yaml"


def save_model(model: lgb.LGBMRegressor, feature_cols: list[str], config: PipelineConfig) -> Path:
    """Persist the LightGBM booster and feature metadata."""
    models_dir = config.path("models_dir")
    ensure_directories([models_dir])

    model_path = model_artifact_path(config)
    metadata_path = metadata_artifact_path(config)
    model.booster_.save_model(str(model_path))
    metadata = {
        "model_name": config.model_name,
        "dataset_version": config.dataset_version,
        "feature_columns": feature_cols,
        "target_col": config.target_col,
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)

    LOGGER.info("Saved model to %s", model_path)
    LOGGER.info("Saved metadata to %s", metadata_path)
    return model_path


def load_model(config: PipelineConfig) -> tuple[lgb.Booster, list[str]]:
    """Load a saved LightGBM booster and feature list."""
    model_path = model_artifact_path(config)
    metadata_path = metadata_artifact_path(config)
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model artifacts not found. Run scripts/train_baseline.py first.")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = yaml.safe_load(file)
    feature_cols = list(metadata["feature_columns"])
    return lgb.Booster(model_file=str(model_path)), feature_cols


def predict_with_model(
    model: lgb.LGBMRegressor | lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    config: PipelineConfig,
) -> pd.DataFrame:
    """Generate normalized Numerai predictions for a dataframe."""
    require_columns(set(df.columns), set(feature_cols) | {config.id_col}, "Prediction data")
    raw_predictions = model.predict(df[feature_cols])
    return make_prediction_frame(
        df[config.id_col],
        raw_predictions,
        id_col=config.id_col,
        prediction_col=config.prediction_col,
    )
