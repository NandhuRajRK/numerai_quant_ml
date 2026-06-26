from __future__ import annotations

import pandas as pd
import pytest

from numerai_quant.features import make_prediction_frame, validate_prediction_frame


def test_make_prediction_frame_normalizes_to_required_columns() -> None:
    frame = make_prediction_frame(
        ids=pd.Series(["a", "b", "c"]),
        predictions=pd.Series([100.0, -10.0, 5.0]),
    )

    assert list(frame.columns) == ["id", "prediction"]
    assert frame["prediction"].between(0.0, 1.0).all()
    assert frame["id"].tolist() == ["a", "b", "c"]


def test_prediction_validation_rejects_out_of_range_values() -> None:
    frame = pd.DataFrame({"id": ["a"], "prediction": [1.2]})

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_prediction_frame(frame)
