import pandas as pd
import pytest

from nce_analysis.config import AnalysisConfig
from nce_analysis.hotspot.ratio_threshold import RatioThreshold


def _long_df():
    rows = []
    for i in range(6):
        rows.append(
            {
                "WaferID": f"W{i}",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0 if i < 4 else 5.0,
            }
        )
    for i in range(6):
        rows.append({"WaferID": f"W{i}", "X_Posi": 6.0, "Y_Posi": 0.0, "NCE_Value": 5.0})
    for i in range(2):
        rows.append({"WaferID": f"S{i}", "X_Posi": 12.0, "Y_Posi": 0.0, "NCE_Value": 20.0})
    return pd.DataFrame(rows)


def test_detect_identifies_hotspot_above_ratio_threshold():
    long_df = _long_df()
    config = AnalysisConfig(spec_threshold=15.0, min_wafer_count=5, hotspot_ratio_threshold=0.05)

    result = RatioThreshold().detect(long_df, config)

    sufficient = result[~result["insufficient_sample"]]
    assert len(sufficient) == 1
    row = sufficient.iloc[0]
    assert (row["X_Posi"], row["Y_Posi"]) == (0.0, 0.0)
    assert row["anomalous_wafer_count"] == 4
    assert row["total_wafer_count"] == 6
    assert row["anomaly_ratio"] == pytest.approx(4 / 6)


def test_detect_flags_coordinate_below_min_wafer_count():
    long_df = _long_df()
    config = AnalysisConfig(min_wafer_count=5)

    result = RatioThreshold().detect(long_df, config)

    flagged = result[(result["X_Posi"] == 12.0) & (result["Y_Posi"] == 0.0)]
    assert len(flagged) == 1
    assert bool(flagged["insufficient_sample"].iloc[0]) is True
    assert flagged["anomaly_ratio"].iloc[0] == pytest.approx(1.0)
