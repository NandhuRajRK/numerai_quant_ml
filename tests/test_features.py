from __future__ import annotations

import numpy as np
import pandas as pd

from numerai_quant.features import apply_feature_controls, prune_correlated_features


def test_prune_correlated_features_drops_redundant_column() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [0, 1, 2, 3, 4, 5],
            "feature_b": [0, 1, 2, 3, 4, 5],
            "feature_c": [0, 1, 0, 1, 0, 1],
        }
    )

    selected, summary = prune_correlated_features(frame, ["feature_a", "feature_b", "feature_c"])

    assert len(selected) == 2
    assert summary["num_dropped"] == 1


def test_apply_feature_controls_limits_feature_count_with_importance_filter() -> None:
    rng = np.random.default_rng(42)
    signal = rng.normal(size=200)
    frame = pd.DataFrame(
        {
            "feature_signal": signal,
            "feature_duplicate": signal,
            "feature_noise_1": rng.normal(size=200),
            "feature_noise_2": rng.normal(size=200),
            "target": signal * 0.8 + rng.normal(scale=0.1, size=200),
        }
    )

    selected, summary = apply_feature_controls(
        frame,
        ["feature_signal", "feature_duplicate", "feature_noise_1", "feature_noise_2"],
        target_col="target",
        config={
            "correlation_pruning": {"enabled": True, "max_abs_corr": 0.995},
            "importance_filtering": {
                "enabled": True,
                "top_n": 2,
                "selector_params": {"n_estimators": 50, "n_jobs": 1, "force_col_wise": True},
            },
        },
    )

    assert len(selected) == 2
    assert summary["num_after_importance_filtering"] == 2
