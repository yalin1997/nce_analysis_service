import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.statistical import StatisticalStrategy


def _group_df_with_significant_combo():
    rows = []
    for i in range(10):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberA", "is_anomaly": i < 9})
    for i in range(10):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberB", "is_anomaly": i < 1})
    for i in range(10):
        rows.append({"Pre_ToolID": "CMP_02", "Pre_ChamberID": "ChamberC", "is_anomaly": i < 1})
    return pd.DataFrame(rows)


def test_analyze_identifies_significant_combo():
    group_df = _group_df_with_significant_combo()
    config = AnalysisConfig(alpha=0.05)

    result = StatisticalStrategy().analyze(group_df, config)

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_01"
    assert result[0].suspect_chamber_id == "ChamberA"
    assert result[0].confidence_score > 90


def test_analyze_can_identify_significant_tool_without_chamber_granularity():
    group_df = _group_df_with_significant_combo()
    config = AnalysisConfig(alpha=0.05, root_cause_granularity="tool")

    result = StatisticalStrategy().analyze(group_df, config)

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_01"
    assert result[0].suspect_chamber_id == "N/A"
    assert result[0].confidence_score > 90


def test_analyze_returns_empty_when_no_significant_association():
    rows = []
    for combo_tool, combo_chamber in [("CMP_01", "ChamberA"), ("CMP_02", "ChamberB")]:
        for i in range(10):
            rows.append(
                {"Pre_ToolID": combo_tool, "Pre_ChamberID": combo_chamber, "is_anomaly": i < 5}
            )
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig(alpha=0.05)

    result = StatisticalStrategy().analyze(group_df, config)

    assert result == []


def test_analyze_uses_fisher_fallback_for_small_expected_counts():
    rows = []
    for _ in range(5):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberA", "is_anomaly": True})
    for _ in range(5):
        rows.append({"Pre_ToolID": "CMP_02", "Pre_ChamberID": "ChamberB", "is_anomaly": False})
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig(alpha=0.05)

    result = StatisticalStrategy().analyze(group_df, config)

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_01"
    assert result[0].metrics["fisher_fallback"] == 1.0
    assert "p_value_global" not in result[0].metrics
