import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.ml import MLStrategy


def _group_df():
    rows = []
    for i in range(30):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberA", "is_anomaly": i < 27})
    for i in range(30):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberB", "is_anomaly": i < 3})
    return pd.DataFrame(rows)


def test_analyze_identifies_top_contributing_combo():
    group_df = _group_df()
    config = AnalysisConfig()

    result = MLStrategy().analyze(group_df, config)

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_01"
    assert result[0].suspect_chamber_id == "ChamberA"
    assert result[0].confidence_score > 0


def test_analyze_can_identify_top_contributing_tool_without_chamber_granularity():
    rows = []
    for i in range(30):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberA", "is_anomaly": i < 27})
    for i in range(30):
        rows.append({"Pre_ToolID": "CMP_02", "Pre_ChamberID": "ChamberB", "is_anomaly": i < 3})
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig(root_cause_granularity="tool")

    result = MLStrategy().analyze(group_df, config)

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_01"
    assert result[0].suspect_chamber_id == "N/A"
    assert result[0].confidence_score > 0


def test_analyze_returns_empty_with_single_combo():
    group_df = pd.DataFrame(
        [
            {"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberA", "is_anomaly": i < 5}
            for i in range(10)
        ]
    )
    config = AnalysisConfig()

    result = MLStrategy().analyze(group_df, config)

    assert result == []


def test_analyze_identifies_best_high_cardinality_combo_without_depth_limit():
    rows = []
    for combo_idx in range(50):
        anomaly_count = 9 if combo_idx == 17 else 1
        for i in range(10):
            rows.append(
                {
                    "Pre_ToolID": f"CMP_{combo_idx:02d}",
                    "Pre_ChamberID": f"Chamber{combo_idx:02d}",
                    "is_anomaly": i < anomaly_count,
                }
            )
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig()

    result = MLStrategy().analyze(group_df, config)

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_17"
    assert result[0].suspect_chamber_id == "Chamber17"
    assert result[0].metrics["risk_uplift"] > 0
    assert result[0].metrics["ml_scoring_method"] == 1.0


def test_analyze_returns_empty_when_no_combo_has_positive_risk_uplift():
    rows = []
    for combo_idx in range(5):
        for i in range(20):
            rows.append(
                {
                    "Pre_ToolID": f"CMP_{combo_idx:02d}",
                    "Pre_ChamberID": f"Chamber{combo_idx:02d}",
                    "is_anomaly": i < 5,
                }
            )
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig()

    result = MLStrategy().analyze(group_df, config)

    assert result == []
