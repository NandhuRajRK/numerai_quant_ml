from __future__ import annotations

import pandas as pd
import pytest

from numerai_quant.metrics import era_correlations, max_drawdown, sharpe_like, spearman_corr


def test_spearman_corr_perfect_rank_order() -> None:
    y_true = pd.Series([0.1, 0.2, 0.3, 0.4])
    y_pred = pd.Series([10.0, 20.0, 30.0, 40.0])

    assert spearman_corr(y_true, y_pred) == 1.0


def test_era_correlations_and_summary_helpers() -> None:
    frame = pd.DataFrame(
        {
            "era": ["era1", "era1", "era2", "era2"],
            "target": [0.1, 0.9, 0.9, 0.1],
            "prediction": [0.2, 0.8, 0.3, 0.7],
        }
    )

    correlations = era_correlations(frame)

    assert correlations.loc["era1"] == pytest.approx(1.0)
    assert correlations.loc["era2"] == pytest.approx(-1.0)
    assert sharpe_like(correlations) == 0.0
    assert max_drawdown(correlations) == pytest.approx(-1.0)
