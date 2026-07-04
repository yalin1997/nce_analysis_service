import pandas as pd
import pytest

from nce_analysis.preprocessing.base import PreprocessingError
from nce_analysis.preprocessing.wide_history_reshape import explode_measurement_points, discover_history_levels


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


def test_discover_history_levels_finds_all_levels():
    columns = [
        "WaferID",
        "X_Posi",
        "Pre_StageID_1",
        "Pre_StepID_1",
        "Pre_ToolID_1",
        "Pre_ChamberID_1",
        "Pre_Execute_Time_1",
        "Pre_StageID_2",
        "Pre_StepID_2",
        "Pre_ToolID_2",
        "Pre_ChamberID_2",
        "Pre_Execute_Time_2",
    ]

    levels = discover_history_levels(columns)

    assert set(levels.keys()) == {1, 2}
    assert levels[1]["StageID"] == "Pre_StageID_1"
    assert levels[1]["ToolID"] == "Pre_ToolID_1"
    assert levels[2]["ChamberID"] == "Pre_ChamberID_2"


def test_discover_history_levels_returns_empty_when_no_match():
    columns = ["WaferID", "X_Posi", "NCE_Value"]

    levels = discover_history_levels(columns)

    assert levels == {}


from nce_analysis.preprocessing.wide_history_reshape import WideHistoryReshape


def _build_raw_df():
    return pd.DataFrame(
        [
            {
                "PartID": "P1",
                "WaferID": "W1",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": "LITHO_TOOL_A",
                "ChuckID": "CHK_1",
                "Execute_Time": "2026-01-01",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 20.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "CMP_STEP",
                "Pre_ToolID_1": "CMP_01",
                "Pre_ChamberID_1": "ChamberA",
                "Pre_Execute_Time_1": "2025-12-31",
                "Pre_StageID_2": "CVD",
                "Pre_StepID_2": "CVD_STEP",
                "Pre_ToolID_2": "CVD_01",
                "Pre_ChamberID_2": "ChamberX",
                "Pre_Execute_Time_2": "2025-12-30",
            },
            {
                "PartID": "P1",
                "WaferID": "W2",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": "LITHO_TOOL_A",
                "ChuckID": "CHK_1",
                "Execute_Time": "2026-01-02",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 5.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "CMP_STEP",
                "Pre_ToolID_1": "CMP_02",
                "Pre_ChamberID_1": "ChamberB",
                "Pre_Execute_Time_1": "2025-12-31",
                "Pre_StageID_2": None,
                "Pre_StepID_2": None,
                "Pre_ToolID_2": None,
                "Pre_ChamberID_2": None,
                "Pre_Execute_Time_2": None,
            },
        ]
    )


def test_transform_expands_history_levels_by_actual_stage_id():
    raw_df = _build_raw_df()

    result = WideHistoryReshape().transform(raw_df)

    assert len(result) == 3
    w1_stages = set(result[result["WaferID"] == "W1"]["Pre_StageID"])
    assert w1_stages == {"CMP", "CVD"}
    w2_rows = result[result["WaferID"] == "W2"]
    assert len(w2_rows) == 1
    assert w2_rows["Pre_StageID"].iloc[0] == "CMP"


def test_transform_raises_when_no_history_columns_present():
    raw_df = pd.DataFrame(
        [{"WaferID": "W1", "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 1.0}]}]
    )

    with pytest.raises(PreprocessingError):
        WideHistoryReshape().transform(raw_df)
