from __future__ import annotations

import pandas as pd

from numerai_quant.validation import (
    apply_feature_neutralization,
    build_walk_forward_folds,
    feature_exposure,
)


def test_build_walk_forward_folds_uses_ordered_eras() -> None:
    frame = pd.DataFrame(
        {
            "era": [f"era{i}" for i in range(1, 13)],
            "target": [0.1] * 12,
            "prediction": [0.1] * 12,
        }
    )

    folds = build_walk_forward_folds(
        frame,
        min_train_eras=4,
        validation_eras=2,
        embargo_eras=1,
        step_size=2,
    )

    assert len(folds) == 3
    assert folds[0].train_eras == ["era1", "era2", "era3", "era4"]
    assert folds[0].validation_eras == ["era6", "era7"]
    assert folds[-1].validation_eras == ["era10", "era11"]


def test_build_walk_forward_folds_reindexes_after_max_fold_trimming() -> None:
    frame = pd.DataFrame(
        {
            "era": [f"era{i}" for i in range(1, 13)],
            "target": [0.1] * 12,
            "prediction": [0.1] * 12,
        }
    )

    folds = build_walk_forward_folds(
        frame,
        min_train_eras=3,
        validation_eras=2,
        embargo_eras=1,
        step_size=1,
        max_folds=3,
    )

    assert [fold.fold_number for fold in folds] == [1, 2, 3]


def test_feature_neutralization_reduces_top_feature_exposure() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [0, 0, 1, 1, 2, 2, 3, 3],
            "feature_b": [0, 1, 0, 1, 0, 1, 0, 1],
            "prediction": [0.0, 0.4, 1.8, 1.2, 3.1, 2.1, 4.0, 3.0],
        }
    )
    before = feature_exposure(frame, ["feature_a", "feature_b"], sample_size=2)
    neutralized = apply_feature_neutralization(
        frame,
        ["feature_a", "feature_b"],
        prediction_col="prediction",
        sample_size=1,
        proportion=1.0,
    )
    after_frame = frame.assign(prediction=neutralized)
    after = feature_exposure(after_frame, ["feature_a", "feature_b"], sample_size=2)

    assert after.loc["feature_a"] < before.loc["feature_a"]


def test_feature_exposure_handles_constant_columns() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [1, 1, 1, 1],
            "feature_b": [0, 1, 0, 1],
            "prediction": [0.2, 0.4, 0.6, 0.8],
        }
    )

    exposures = feature_exposure(frame, ["feature_a", "feature_b"], sample_size=2)

    assert exposures.loc["feature_a"] == 0.0
