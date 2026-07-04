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
