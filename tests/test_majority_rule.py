import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.noise_filter.majority_rule import MajorityRule


def _hotspots(coords):
    return pd.DataFrame(coords, columns=["X_Posi", "Y_Posi"])


def test_chuck_majority_single_stage_classified_as_chuck_issue():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": f"W{i}",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "StageID": "LITHO_M1",
            }
            for i in range(6)
        ]
    )
    hotspots = _hotspots([(0.0, 0.0)])
    config = AnalysisConfig(noise_filter_majority_threshold=0.5)

    result = MajorityRule().filter(long_df, hotspots, config)

    assert len(result.surviving_hotspots) == 0
    assert len(result.litho_self_issues) == 1
    assert result.litho_self_issues[0].Root_Cause_Type == "LITHO_CHUCK_ISSUE"
    assert result.litho_self_issues[0].Suspect_Pre_ChamberID == "CHK_1"


def test_chuck_majority_multi_stage_classified_as_contamination():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": f"W{i}",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "StageID": "LITHO_M1" if i % 2 == 0 else "LITHO_M2",
            }
            for i in range(6)
        ]
    )
    hotspots = _hotspots([(0.0, 0.0)])
    config = AnalysisConfig(noise_filter_majority_threshold=0.5)

    result = MajorityRule().filter(long_df, hotspots, config)

    assert result.litho_self_issues[0].Root_Cause_Type == "LITHO_CHUCK_CONTAMINATION"


def test_tool_majority_without_chuck_majority_classified_as_tool_issue():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": f"W{i}",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": f"CHK_{i % 3}",
                "StageID": "LITHO_M1",
            }
            for i in range(6)
        ]
    )
    hotspots = _hotspots([(0.0, 0.0)])
    config = AnalysisConfig(noise_filter_majority_threshold=0.5)

    result = MajorityRule().filter(long_df, hotspots, config)

    assert len(result.surviving_hotspots) == 0
    assert result.litho_self_issues[0].Root_Cause_Type == "LITHO_TOOL_ISSUE"
    assert result.litho_self_issues[0].Suspect_Pre_ToolID == "LITHO_A"


def test_no_majority_survives_to_root_cause_analysis():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": f"W{i}",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": f"LITHO_{i % 3}",
                "ChuckID": f"CHK_{i % 3}",
                "StageID": "LITHO_M1",
            }
            for i in range(6)
        ]
    )
    hotspots = _hotspots([(0.0, 0.0)])
    config = AnalysisConfig(noise_filter_majority_threshold=0.5)

    result = MajorityRule().filter(long_df, hotspots, config)

    assert len(result.surviving_hotspots) == 1
    assert len(result.litho_self_issues) == 0
