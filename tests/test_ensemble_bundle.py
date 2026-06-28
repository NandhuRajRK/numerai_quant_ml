from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from numerai_quant.config import load_config, load_config_mapping
from numerai_quant.ensemble import (
    enabled_model_specs,
    fit_model,
    load_ensemble_bundle,
    save_ensemble_bundle,
)


def test_save_ensemble_bundle_persists_metadata(tmp_path: Path) -> None:
    base_config = load_config("configs/local_smoke.yaml")
    raw = deepcopy(base_config.raw)
    raw["paths"]["models_dir"] = "models"
    config = load_config_mapping(raw, root_dir=tmp_path)

    features = ["feature_a", "feature_b"]
    training_frame = {
        "feature_a": [0.1, 0.2, 0.8, 0.9, 0.15, 0.25, 0.75, 0.85],
        "feature_b": [0.9, 0.7, 0.4, 0.2, 0.8, 0.6, 0.3, 0.1],
        "target": [0.0, 0.1, 0.9, 1.0, 0.05, 0.2, 0.8, 0.95],
    }

    import pandas as pd

    df = pd.DataFrame(training_frame)
    specs = enabled_model_specs(config)
    models = {
        spec["name"]: fit_model(spec, df[features], df["target"])
        for spec in specs
    }

    bundle_dir = save_ensemble_bundle(models, features, config, artifact_name="test_bundle")
    metadata_path = bundle_dir / "metadata.yaml"

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = yaml.safe_load(file)

    assert metadata["artifact_name"] == "test_bundle"
    assert metadata["feature_columns"] == features
    assert metadata["id_col"] == config.id_col
    assert metadata["prediction_col"] == config.prediction_col
    assert [spec["name"] for spec in metadata["model_specs"]] == ["lgbm_main", "lgbm_alt"]

    loaded_models, loaded_metadata = load_ensemble_bundle(bundle_dir)

    assert set(loaded_models) == {"lgbm_main", "lgbm_alt"}
    assert loaded_metadata["weights"]["lgbm_main"] == 0.5
