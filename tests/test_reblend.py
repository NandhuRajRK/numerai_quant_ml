from __future__ import annotations

import pandas as pd

from numerai_quant.config import load_config
from numerai_quant.reblend import optimize_cached_blend_weights


def test_optimize_cached_blend_weights_prefers_stronger_model() -> None:
    config = load_config("configs/local_smoke.yaml")
    checkpoint = pd.DataFrame(
        {
            "id": [f"row_{index}" for index in range(8)],
            "era": ["era1"] * 4 + ["era2"] * 4,
            "target": [0.1, 0.2, 0.8, 0.9, 0.15, 0.25, 0.75, 0.85],
            "good_model_prediction": [0.1, 0.2, 0.8, 0.9, 0.15, 0.25, 0.75, 0.85],
            "bad_model_prediction": [0.6, 0.1, 0.3, 0.7, 0.55, 0.2, 0.4, 0.8],
        }
    )

    weights, leaderboard = optimize_cached_blend_weights(
        [checkpoint],
        ["good_model", "bad_model"],
        config=config,
        objective="mean_corr",
        grid_step=0.5,
    )

    assert weights["good_model"] > weights["bad_model"]
    assert float(leaderboard.iloc[0]["mean_corr"]) >= 0.99


def test_optimize_cached_blend_weights_returns_ranked_candidates() -> None:
    config = load_config("configs/local_smoke.yaml")
    checkpoint = pd.DataFrame(
        {
            "id": [f"row_{index}" for index in range(6)],
            "era": ["era1"] * 3 + ["era2"] * 3,
            "target": [0.1, 0.5, 0.9, 0.2, 0.6, 0.8],
            "model_a_prediction": [0.2, 0.4, 0.8, 0.1, 0.5, 0.7],
            "model_b_prediction": [0.3, 0.6, 0.7, 0.2, 0.4, 0.9],
            "model_c_prediction": [0.9, 0.2, 0.1, 0.8, 0.3, 0.4],
        }
    )

    weights, leaderboard = optimize_cached_blend_weights(
        [checkpoint],
        ["model_a", "model_b", "model_c"],
        config=config,
        objective="sharpe_like",
        grid_step=0.5,
    )

    assert set(weights) == {"model_a", "model_b", "model_c"}
    assert "objective_value" in leaderboard.columns
    assert int(leaderboard.iloc[0]["candidate_rank"]) == 1
