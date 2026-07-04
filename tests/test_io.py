import json

import pandas as pd
import pytest

from nce_analysis.io import InputFormatError, load_input


def _wafer_row(wafer_id, **overrides):
    row = {
        "PartID": "PART1",
        "WaferID": wafer_id,
        "StageID": "LITHO_M1",
        "StepID": "S1",
        "ToolID": "LITHO_A",
        "ChuckID": "CHK_1",
        "Execute_Time": "2026-01-01",
        "Pre_StageID_1": "CMP",
        "Pre_StepID_1": "3580.01",
        "Pre_ToolID_1": "CMP_01",
        "Pre_ChamberID_1": "ChamberA",
        "Pre_Execute_Time_1": "2025-12-01",
    }
    row.update(overrides)
    return row


def test_load_input_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("not a real data file")

    with pytest.raises(InputFormatError):
        load_input(path)


def test_load_input_csv_wide_format(tmp_path):
    row = _wafer_row("W1")
    row["Measurement_Points"] = json.dumps(
        [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0}]
    )
    csv_path = tmp_path / "data.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    result = load_input(csv_path)

    assert len(result) == 1
    points = result.iloc[0]["Measurement_Points"]
    assert points == [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0}]


def test_load_input_csv_long_format_groups_points_by_wafer(tmp_path):
    rows = [
        {**_wafer_row("W1"), "X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0},
        {**_wafer_row("W1"), "X_Posi": 6.0, "Y_Posi": 0.0, "NCE_Value": 5.0},
        {**_wafer_row("W2"), "X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 20.0},
    ]
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = load_input(csv_path)

    assert len(result) == 2
    w1_points = result.loc[result["WaferID"] == "W1", "Measurement_Points"].iloc[0]
    assert len(w1_points) == 2
    assert {"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0} in w1_points


def test_load_input_csv_missing_wafer_id_raises(tmp_path):
    rows = [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0}]
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with pytest.raises(InputFormatError):
        load_input(csv_path)


def test_load_input_csv_unrecognized_shape_raises(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame([{"foo": 1, "bar": 2}]).to_csv(csv_path, index=False)

    with pytest.raises(InputFormatError):
        load_input(csv_path)


def test_load_input_parquet_round_trips_measurement_points_as_list(tmp_path):
    row = _wafer_row("W1")
    row["Measurement_Points"] = [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0}]
    parquet_path = tmp_path / "data.parquet"
    pd.DataFrame([row]).to_parquet(parquet_path)

    result = load_input(parquet_path)

    points = result.iloc[0]["Measurement_Points"]
    assert isinstance(points, list)
    assert points == [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0}]
