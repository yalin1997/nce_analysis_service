import pandas as pd

from nce_analysis import pipeline
from nce_analysis.config import AnalysisConfig


def _build_batch():
    rows = []
    litho_tools = ["LITHO_A", "LITHO_B", "LITHO_C"]
    litho_chucks = ["CHK_1", "CHK_2", "CHK_3"]

    for i in range(18):
        rows.append(
            {
                "PartID": "PART1",
                "WaferID": f"ANOM_{i}",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": litho_tools[i % 3],
                "ChuckID": litho_chucks[i % 3],
                "Execute_Time": f"2026-01-{(i % 28) + 1:02d}",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 25.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "CMP_STEP",
                "Pre_ToolID_1": "CMP_01",
                "Pre_ChamberID_1": "ChamberA",
                "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
            }
        )
    for i in range(6):
        rows.append(
            {
                "PartID": "PART1",
                "WaferID": f"NORM_B_{i}",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": litho_tools[i % 3],
                "ChuckID": litho_chucks[i % 3],
                "Execute_Time": f"2026-02-{(i % 28) + 1:02d}",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 5.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "CMP_STEP",
                "Pre_ToolID_1": "CMP_01",
                "Pre_ChamberID_1": "ChamberB",
                "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
            }
        )
    for i in range(6):
        rows.append(
            {
                "PartID": "PART1",
                "WaferID": f"NORM_C_{i}",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": litho_tools[i % 3],
                "ChuckID": litho_chucks[i % 3],
                "Execute_Time": f"2026-03-{(i % 28) + 1:02d}",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 5.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "CMP_STEP",
                "Pre_ToolID_1": "CMP_02",
                "Pre_ChamberID_1": "ChamberC",
                "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
            }
        )
    return pd.DataFrame(rows)


def test_pipeline_identifies_injected_root_cause():
    raw_df = _build_batch()
    config = AnalysisConfig(root_cause_strategy="statistical", drift_strategy="regression_cusum")

    result = pipeline.run(raw_df, config)

    assert len(result.Details) >= 1
    top = result.Summary[0]
    assert top.Suspect_Pre_ToolID == "CMP_01"
    assert top.Suspect_Pre_ChamberID == "ChamberA"
    assert top.Suspect_Pre_StepID == "CMP_STEP"
    assert top.Root_Cause_Type in {
        "SPECIFIC_CHAMBER_DEFECT",
        "CHAMBER_DRIFT",
        "CHAMBER_SUDDEN_SHIFT",
    }
    assert top.Requires_Manual_Review is False


def test_pipeline_returns_empty_result_when_no_hotspots():
    raw_df = pd.DataFrame(
        [
            {
                "PartID": "P1",
                "WaferID": "W1",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "Execute_Time": "2026-01-01",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 5.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "CMP_STEP",
                "Pre_ToolID_1": "CMP_01",
                "Pre_ChamberID_1": "ChamberA",
                "Pre_Execute_Time_1": "2025-12-01",
            }
        ]
    )
    config = AnalysisConfig(min_wafer_count=5)

    result = pipeline.run(raw_df, config)

    assert result.Details == []
    assert result.Summary == []
    assert result.Insufficient_Sample_Hotspots == []


def test_pipeline_distinguishes_repeated_stage_visits_by_step_id():
    rows = []
    litho_tools = ["LITHO_A", "LITHO_B", "LITHO_C"]
    litho_chucks = ["CHK_1", "CHK_2", "CHK_3"]
    for i in range(24):
        anomalous = i < 12
        rows.append(
            {
                "PartID": "PART1",
                "WaferID": f"W{i}",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": litho_tools[i % 3],
                "ChuckID": litho_chucks[i % 3],
                "Execute_Time": f"2026-01-{(i % 28) + 1:02d}",
                "Measurement_Points": [
                    {"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 25.0 if anomalous else 5.0}
                ],
                # Second CMP pass (most recent history level): tool assignment
                # balanced across anomaly status — must NOT be blamed.
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "3580.02",
                "Pre_ToolID_1": "CMP_C" if i % 2 == 0 else "CMP_D",
                "Pre_ChamberID_1": "ChamberM" if i % 2 == 0 else "ChamberN",
                "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
                # First CMP pass (earlier history level): the injected culprit.
                "Pre_StageID_2": "CMP",
                "Pre_StepID_2": "3580.01",
                "Pre_ToolID_2": "CMP_A" if anomalous else "CMP_B",
                "Pre_ChamberID_2": "ChamberX" if anomalous else "ChamberY",
                "Pre_Execute_Time_2": f"2025-11-{(i % 28) + 1:02d}",
            }
        )
    raw_df = pd.DataFrame(rows)
    config = AnalysisConfig(root_cause_strategy="statistical")

    result = pipeline.run(raw_df, config)

    top = result.Summary[0]
    assert top.Suspect_Pre_ToolID == "CMP_A"
    assert top.Suspect_Pre_ChamberID == "ChamberX"
    assert top.Suspect_Pre_StepID == "3580.01"
    assert not any(d.Suspect_Pre_StepID == "3580.02" for d in result.Details)


def test_pipeline_reports_insufficient_sample_hotspot_without_confidence():
    rows = []
    for i in range(2):
        rows.append(
            {
                "PartID": "P1",
                "WaferID": f"W{i}",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "Execute_Time": f"2026-01-0{i + 1}",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 25.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "3580.01",
                "Pre_ToolID_1": "CMP_01",
                "Pre_ChamberID_1": "ChamberA",
                "Pre_Execute_Time_1": f"2025-12-0{i + 1}",
            }
        )
    raw_df = pd.DataFrame(rows)
    config = AnalysisConfig(min_wafer_count=5)

    result = pipeline.run(raw_df, config)

    assert result.Details == []
    assert result.Summary == []
    assert len(result.Insufficient_Sample_Hotspots) == 1
    hotspot = result.Insufficient_Sample_Hotspots[0]
    assert (hotspot.X_Posi, hotspot.Y_Posi) == (0.0, 0.0)
    assert hotspot.anomaly_ratio == 1.0
    assert hotspot.total_wafer_count == 2
