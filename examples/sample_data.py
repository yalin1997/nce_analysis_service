"""Synthetic wafer batch for trying out nce_analysis.pipeline.run().

Shape matches the raw input contract (see spec §2): one row per wafer, with
Measurement_Points holding the surface-flatness readings and Pre_*_<N>
columns holding the upstream process history. This batch packs three
scenarios into one small DataFrame so a single pipeline.run() call exercises
the whole decision tree:

- (X=0.0, Y=0.0): 15 wafers processed by a drifting CMP chamber (CMP_01 /
  ChamberA) score high NCE, worsening over time; 5 contrast wafers processed
  elsewhere score normally. Root-cause analysis should pin this on
  CMP_01/ChamberA with Root_Cause_Type "CHAMBER_DRIFT".
- (X=6.0, Y=0.0): only 3 wafers total, below the default min_wafer_count (5).
  Reported in Insufficient_Sample_Hotspots with no confidence score.
- (X=12.0, Y=0.0): 8 anomalous wafers share one LITHO chuck (CHK_9) despite
  varied upstream tool/chamber. The noise filter should attribute this to
  the chuck itself ("LITHO_CHUCK_ISSUE") and exclude it from upstream
  root-cause analysis.
"""

import pandas as pd


def build_sample_dataframe() -> pd.DataFrame:
    rows = []

    # --- Group A: drifting CMP chamber causes a hotspot at (0.0, 0.0) ---
    # NCE_Value climbs as Pre_Execute_Time_1 (when CMP processed the wafer)
    # gets later, so RegressionCusum's slope test should flag CHAMBER_DRIFT.
    litho_tools = ["LITHO_TOOL_1", "LITHO_TOOL_4", "LITHO_TOOL_5"]
    litho_chucks = ["CHK_1", "CHK_2", "CHK_3", "CHK_4"]
    for i in range(15):
        jitter = 0.4 if i % 2 == 0 else -0.4
        rows.append({
            "PartID": "PART_A",
            "WaferID": f"WAFER_A{i:02d}",
            "StageID": "LITHO_M1",
            "StepID": "S100",
            "ToolID": litho_tools[i % 3],
            "ChuckID": litho_chucks[i % 4],
            "Execute_Time": f"2025-12-{(i % 28) + 1:02d}",
            "Measurement_Points": [
                {"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0 + i + jitter}
            ],
            "Pre_StageID_1": "CMP",
            "Pre_StepID_1": "3580.01",
            "Pre_ToolID_1": "CMP_01",
            "Pre_ChamberID_1": "ChamberA",
            "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
            # Second history level (CVD): identical for every Group A wafer,
            # so it carries no discriminating signal -- demonstrates that
            # grouping by the *actual* (Pre_StageID, Pre_StepID) correctly
            # isolates the CMP suspect instead of blaming CVD too.
            "Pre_StageID_2": "CVD",
            "Pre_StepID_2": "2100.01",
            "Pre_ToolID_2": "CVD_01",
            "Pre_ChamberID_2": "ChamberX",
            "Pre_Execute_Time_2": "2025-11-15",
        })

    # --- Group A-normal: 5 contrast wafers at the same coordinate ---
    for i in range(5):
        rows.append({
            "PartID": "PART_A",
            "WaferID": f"WAFER_N{i:02d}",
            "StageID": "LITHO_M1",
            "StepID": "S100",
            "ToolID": "LITHO_TOOL_1",
            "ChuckID": "CHK_1",
            "Execute_Time": f"2025-12-{(i % 28) + 1:02d}",
            "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 5.0}],
            "Pre_StageID_1": "CMP",
            "Pre_StepID_1": "3580.01",
            "Pre_ToolID_1": "CMP_02",
            "Pre_ChamberID_1": "ChamberC",
            "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
            "Pre_StageID_2": "CVD",
            "Pre_StepID_2": "2100.01",
            "Pre_ToolID_2": "CVD_01",
            "Pre_ChamberID_2": "ChamberX",
            "Pre_Execute_Time_2": "2025-11-15",
        })

    # --- Group B: 3 wafers at (6.0, 0.0) -> below min_wafer_count ---
    for i, nce in enumerate([19.0, 21.0, 20.0]):
        rows.append({
            "PartID": "PART_B",
            "WaferID": f"WAFER_B{i:02d}",
            "StageID": "LITHO_M1",
            "StepID": "S200",
            "ToolID": "LITHO_TOOL_1",
            "ChuckID": "CHK_2",
            "Execute_Time": f"2025-12-{(i % 28) + 1:02d}",
            "Measurement_Points": [{"X_Posi": 6.0, "Y_Posi": 0.0, "NCE_Value": nce}],
            "Pre_StageID_1": "CMP",
            "Pre_StepID_1": "3580.01",
            "Pre_ToolID_1": "CMP_04",
            "Pre_ChamberID_1": "ChamberE",
            "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
            "Pre_StageID_2": None,
            "Pre_StepID_2": None,
            "Pre_ToolID_2": None,
            "Pre_ChamberID_2": None,
            "Pre_Execute_Time_2": None,
        })

    # --- Group C: 8 anomalous wafers at (12.0, 0.0) sharing LITHO chuck
    # CHK_9 despite varied upstream tool/chamber -> noise filter should
    # attribute this to the chuck, not to any upstream combo.
    upstream_variants = [("CMP_01", "ChamberA"), ("CMP_02", "ChamberB"), ("CMP_03", "ChamberD")]
    for i in range(8):
        tool, chamber = upstream_variants[i % 3]
        rows.append({
            "PartID": "PART_C",
            "WaferID": f"WAFER_C{i:02d}",
            "StageID": "LITHO_M1",
            "StepID": "S300",
            "ToolID": "LITHO_TOOL_2" if i % 2 == 0 else "LITHO_TOOL_3",
            "ChuckID": "CHK_9",
            "Execute_Time": f"2025-12-{(i % 28) + 1:02d}",
            "Measurement_Points": [{"X_Posi": 12.0, "Y_Posi": 0.0, "NCE_Value": 20.0}],
            "Pre_StageID_1": "CMP",
            "Pre_StepID_1": "3580.01",
            "Pre_ToolID_1": tool,
            "Pre_ChamberID_1": chamber,
            "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
            "Pre_StageID_2": None,
            "Pre_StepID_2": None,
            "Pre_ToolID_2": None,
            "Pre_ChamberID_2": None,
            "Pre_Execute_Time_2": None,
        })

    # --- Group C-normal: 2 contrast wafers at the same coordinate, different chuck ---
    for i, chuck in enumerate(["CHK_10", "CHK_11"]):
        rows.append({
            "PartID": "PART_C",
            "WaferID": f"WAFER_C_OK{i:02d}",
            "StageID": "LITHO_M1",
            "StepID": "S300",
            "ToolID": "LITHO_TOOL_2",
            "ChuckID": chuck,
            "Execute_Time": f"2025-12-{(i % 28) + 1:02d}",
            "Measurement_Points": [{"X_Posi": 12.0, "Y_Posi": 0.0, "NCE_Value": 5.0}],
            "Pre_StageID_1": "CMP",
            "Pre_StepID_1": "3580.01",
            "Pre_ToolID_1": "CMP_01",
            "Pre_ChamberID_1": "ChamberA",
            "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
            "Pre_StageID_2": None,
            "Pre_StepID_2": None,
            "Pre_ToolID_2": None,
            "Pre_ChamberID_2": None,
            "Pre_Execute_Time_2": None,
        })

    return pd.DataFrame(rows)
