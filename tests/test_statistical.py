import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.statistical import StatisticalStrategy


def _group_df_with_significant_combo():
    # ChamberA is the clear per-chamber suspect (19/20), ChamberB is elevated
    # but weaker (17/20) so that Holm-Bonferroni correction across the 3
    # chamber-level hypotheses still unambiguously singles out ChamberA,
    # while their combined tool-level rate (36/40 vs. ChamberC's 2/20) stays
    # far past the alpha bar even after Holm correction across the 2
    # tool-level hypotheses.
    rows = []
    for i in range(20):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberA", "is_anomaly": i < 19})
    for i in range(20):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberB", "is_anomaly": i < 17})
    for i in range(20):
        rows.append({"Pre_ToolID": "CMP_02", "Pre_ChamberID": "ChamberC", "is_anomaly": i < 2})
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


def test_analyze_rejects_raw_significant_combo_after_holm_correction():
    rows = []
    for combo_idx in range(20):
        anomaly_count = 8 if combo_idx == 0 else 3
        for i in range(20):
            rows.append(
                {
                    "Pre_ToolID": f"CMP_{combo_idx:02d}",
                    "Pre_ChamberID": f"Chamber{combo_idx:02d}",
                    "is_anomaly": i < anomaly_count,
                }
            )
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig(alpha=0.05)

    result = StatisticalStrategy().analyze(group_df, config)

    assert result == []


def test_analyze_keeps_strong_combo_after_holm_correction():
    rows = []
    for combo_idx in range(20):
        anomaly_count = 18 if combo_idx == 0 else 2
        for i in range(20):
            rows.append(
                {
                    "Pre_ToolID": f"CMP_{combo_idx:02d}",
                    "Pre_ChamberID": f"Chamber{combo_idx:02d}",
                    "is_anomaly": i < anomaly_count,
                }
            )
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig(alpha=0.05)

    result = StatisticalStrategy().analyze(group_df, config)

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_00"
    assert result[0].suspect_chamber_id == "Chamber00"
    assert result[0].metrics["p_value_combo_adjusted"] < config.alpha
    assert result[0].metrics["multiple_testing_method"] == 1.0
