"""Model zoo, blending, and persistence for portfolio-grade Numerai experiments."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from numerai_quant.config import PipelineConfig
from numerai_quant.features import make_prediction_frame
from numerai_quant.mlx_models import MLXTabularMLPRegressor
from numerai_quant.utils import ensure_directories, require_columns

LOGGER = logging.getLogger(__name__)


def enabled_model_specs(config: PipelineConfig) -> list[dict[str, Any]]:
    """Return enabled model specifications from the advanced config."""
    raw_specs = config.raw["advanced"]["model_zoo"]
    enabled = [spec for spec in raw_specs if spec.get("enabled", True)]
    if not enabled:
        raise ValueError("No enabled models found in advanced.model_zoo.")
    return enabled


def normalized_weights(specs: list[dict[str, Any]]) -> dict[str, float]:
    """Return normalized ensemble weights."""
    weights = {spec["name"]: float(spec.get("weight", 1.0)) for spec in specs}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    return {name: value / total for name, value in weights.items()}


def _xgboost_regressor() -> Any:
    """Import XGBoost lazily so the project still works without the extra."""
    module = importlib.import_module("xgboost")
    return module.XGBRegressor


def _catboost_regressor() -> Any:
    """Import CatBoost lazily so the project still works without the extra."""
    module = importlib.import_module("catboost")
    return module.CatBoostRegressor


def _mlx_mlp_regressor() -> type[MLXTabularMLPRegressor]:
    """Return the MLX MLP wrapper class."""
    return MLXTabularMLPRegressor


def create_model(spec: dict[str, Any]) -> Any:
    """Instantiate a model from a config spec."""
    model_type = spec["type"]
    params = dict(spec["params"])
    if model_type == "lightgbm":
        return lgb.LGBMRegressor(**params)
    if model_type == "xgboost":
        return _xgboost_regressor()(**params)
    if model_type == "catboost":
        return _catboost_regressor()(**params)
    if model_type == "mlx_mlp":
        return _mlx_mlp_regressor()(**params)
    raise ValueError(f"Unsupported model type: {model_type}")


def fit_model(
    spec: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    **fit_kwargs: Any,
) -> Any:
    """Fit a configured model."""
    model = create_model(spec)
    LOGGER.info("Training %s (%s)", spec["name"], spec["type"])
    if spec["type"] == "mlx_mlp":
        model.fit(X, y, **fit_kwargs)
    else:
        model.fit(X, y)
    return model


def raw_predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return raw predictions from a trained model."""
    predictions = model.predict(X)
    return np.asarray(predictions, dtype=float)


def blend_predictions(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Blend raw predictions using normalized weights."""
    if not predictions:
        raise ValueError("No predictions were provided for blending.")
    output = np.zeros(len(next(iter(predictions.values()))), dtype=float)
    for name, values in predictions.items():
        output += weights[name] * values
    return output


def save_model_file(model: Any, spec: dict[str, Any], model_path: Path) -> Path:
    """Persist a trained model to disk."""
    if spec["type"] == "lightgbm":
        model.booster_.save_model(str(model_path))
    elif spec["type"] == "xgboost":
        model.save_model(str(model_path))
    elif spec["type"] == "catboost":
        model.save_model(str(model_path))
    elif spec["type"] == "mlx_mlp":
        model.save_model(str(model_path))
    else:  # pragma: no cover - guarded by create_model
        raise ValueError(f"Unsupported model type: {spec['type']}")
    return model_path


def ensemble_bundle_dir(config: PipelineConfig, artifact_name: str | None = None) -> Path:
    """Return the directory containing the latest saved ensemble bundle."""
    bundle_name = artifact_name or str(config.raw["advanced"]["artifact_name"])
    return config.path("models_dir") / bundle_name


def save_ensemble_bundle(
    models: dict[str, Any],
    feature_cols: list[str],
    config: PipelineConfig,
    *,
    artifact_name: str | None = None,
) -> Path:
    """Persist a trained ensemble and metadata for live prediction."""
    specs = enabled_model_specs(config)
    weights = normalized_weights(specs)
    bundle_dir = ensemble_bundle_dir(config, artifact_name)
    ensure_directories([bundle_dir])

    metadata = {
        "artifact_name": artifact_name or str(config.raw["advanced"]["artifact_name"]),
        "dataset_version": config.dataset_version,
        "feature_columns": feature_cols,
        "prediction_col": config.prediction_col,
        "id_col": config.id_col,
        "model_specs": [],
        "neutralization": dict(config.raw["advanced"]["neutralization"]),
        "weights": weights,
    }
    for spec in specs:
        extension_map = {
            "lightgbm": "txt",
            "xgboost": "json",
            "catboost": "cbm",
            "mlx_mlp": "mlx",
        }
        extension = extension_map[spec["type"]]
        model_filename = f"{spec['name']}.{extension}"
        save_model_file(models[spec["name"]], spec, bundle_dir / model_filename)
        metadata["model_specs"].append(
            {
                "name": spec["name"],
                "type": spec["type"],
                "weight": weights[spec["name"]],
                "params": dict(spec["params"]),
                "filename": model_filename,
            }
        )

    with (bundle_dir / "metadata.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)
    return bundle_dir


def load_ensemble_bundle(bundle_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a saved ensemble bundle and metadata."""
    metadata_path = bundle_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing ensemble metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = yaml.safe_load(file)

    models: dict[str, Any] = {}
    for spec in metadata["model_specs"]:
        model_path = bundle_dir / spec["filename"]
        if spec["type"] == "lightgbm":
            models[spec["name"]] = lgb.Booster(model_file=str(model_path))
        elif spec["type"] == "xgboost":
            model = _xgboost_regressor()()
            model.load_model(str(model_path))
            models[spec["name"]] = model
        elif spec["type"] == "catboost":
            model = _catboost_regressor()()
            model.load_model(str(model_path))
            models[spec["name"]] = model
        elif spec["type"] == "mlx_mlp":
            models[spec["name"]] = _mlx_mlp_regressor().load_model(str(model_path))
        else:  # pragma: no cover - guarded by metadata generation
            raise ValueError(f"Unsupported model type in metadata: {spec['type']}")
    return models, metadata


def predict_with_ensemble_bundle(
    models: dict[str, Any],
    metadata: dict[str, Any],
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Generate blended predictions from a loaded ensemble bundle."""
    feature_cols = list(metadata["feature_columns"])
    id_col = str(metadata["id_col"])
    prediction_col = str(metadata["prediction_col"])
    require_columns(set(df.columns), set(feature_cols) | {id_col}, "Prediction data")

    raw_by_model = {}
    for spec in metadata["model_specs"]:
        raw_by_model[spec["name"]] = raw_predict(models[spec["name"]], df[feature_cols])

    weights = {spec["name"]: float(spec["weight"]) for spec in metadata["model_specs"]}
    blended = blend_predictions(raw_by_model, weights)
    frame = make_prediction_frame(df[id_col], blended, id_col=id_col, prediction_col=prediction_col)
    return frame, raw_by_model
