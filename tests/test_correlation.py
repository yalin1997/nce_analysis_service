import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.correlation import Correlation


def _timestamps(n):
    return pd.date_range("2026-01-01", periods=n, freq="h")


def test_classify_detects_drift_via_correlation():
    n = 20
    series_df = pd.DataFrame(
        {"Pre_Execute_Time": _timestamps(n), "NCE_Value": [10.0 + 0.5 * i for i in range(n)]}
    )
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, metrics = Correlation().classify(series_df, config)

    assert root_cause_type == "CHAMBER_DRIFT"
    assert metrics["correlation"] > 0.9


def test_classify_returns_stable_when_uncorrelated():
    n = 21
    pattern = [20.0, 20.2, 19.8]
    values = (pattern * (n // 3 + 1))[:n]
    series_df = pd.DataFrame({"Pre_Execute_Time": _timestamps(n), "NCE_Value": values})
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, _ = Correlation().classify(series_df, config)

    assert root_cause_type == "SPECIFIC_CHAMBER_DEFECT"


def test_classify_handles_constant_timestamps_without_nan_metrics():
    n = 5
    same_time = pd.Timestamp("2026-01-01")
    series_df = pd.DataFrame(
        {"Pre_Execute_Time": [same_time] * n, "NCE_Value": [10.0, 12.0, 11.0, 13.0, 9.0]}
    )
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, metrics = Correlation().classify(series_df, config)

    assert root_cause_type == "SPECIFIC_CHAMBER_DEFECT"
    assert metrics["insufficient_time_variation"] == 1.0
    assert not any(value != value for value in metrics.values())


def test_classify_handles_constant_values_without_nan_metrics():
    n = 5
    series_df = pd.DataFrame({"Pre_Execute_Time": _timestamps(n), "NCE_Value": [15.0] * n})
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, metrics = Correlation().classify(series_df, config)

    assert root_cause_type == "SPECIFIC_CHAMBER_DEFECT"
    assert metrics["insufficient_value_variation"] == 1.0
    assert not any(value != value for value in metrics.values())
