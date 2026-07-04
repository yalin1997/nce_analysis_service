import pandas as pd
import pytest

from nce_analysis.preprocessing.base import PreprocessingError
from nce_analysis.preprocessing.wide_history_reshape import explode_measurement_points


def test_explode_measurement_points_expands_rows():
    raw_df = pd.DataFrame(
        [
            {
                "WaferID": "W1",
                "ToolID": "LITHO_01",
                "Measurement_Points": [
                    {"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 10.0},
                    {"X_Posi": 6.0, "Y_Posi": 0.0, "NCE_Value": 20.0},
                ],
            },
            {
                "WaferID": "W2",
                "ToolID": "LITHO_02",
                "Measurement_Points": [
                    {"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 5.0},
                ],
            },
        ]
    )

    result = explode_measurement_points(raw_df)

    assert len(result) == 3
    assert set(result.columns) == {"WaferID", "ToolID", "X_Posi", "Y_Posi", "NCE_Value"}
    assert result.loc[result["WaferID"] == "W2", "NCE_Value"].iloc[0] == 5.0
    assert (result["WaferID"] == "W1").sum() == 2


def test_explode_skips_wafers_with_empty_measurement_points(caplog):
    raw_df = pd.DataFrame(
        [
            {
                "WaferID": "W1",
                "ToolID": "LITHO_01",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 10.0}],
            },
            {"WaferID": "W2", "ToolID": "LITHO_02", "Measurement_Points": []},
        ]
    )

    with caplog.at_level("WARNING"):
        result = explode_measurement_points(raw_df)

    assert result["WaferID"].tolist() == ["W1"]
    assert "W2" in caplog.text


def test_explode_raises_when_all_wafers_have_empty_measurement_points():
    raw_df = pd.DataFrame([{"WaferID": "W1", "Measurement_Points": []}])

    with pytest.raises(PreprocessingError):
        explode_measurement_points(raw_df)
