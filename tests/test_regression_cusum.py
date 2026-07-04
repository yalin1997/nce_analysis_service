import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.regression_cusum import RegressionCusum


def _timestamps(n):
    return pd.date_range("2026-01-01", periods=n, freq="h")


def test_classify_detects_gradual_drift():
    n = 20
    series_df = pd.DataFrame(
        {"Pre_Execute_Time": _timestamps(n), "NCE_Value": [10.0 + 0.5 * i for i in range(n)]}
    )
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, metrics = RegressionCusum().classify(series_df, config)

    assert root_cause_type == "CHAMBER_DRIFT"
    assert metrics["slope"] > 0


def test_classify_detects_sudden_shift():
    n = 20
    values = [10.0] * (n // 2) + [30.0] * (n // 2)
    series_df = pd.DataFrame({"Pre_Execute_Time": _timestamps(n), "NCE_Value": values})
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, _ = RegressionCusum().classify(series_df, config)

    assert root_cause_type == "CHAMBER_SUDDEN_SHIFT"


def test_classify_detects_stable_defect_with_no_trend():
    n = 21
    pattern = [20.0, 20.2, 19.8]
    values = (pattern * (n // 3 + 1))[:n]
    series_df = pd.DataFrame({"Pre_Execute_Time": _timestamps(n), "NCE_Value": values})
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, _ = RegressionCusum().classify(series_df, config)

    assert root_cause_type == "SPECIFIC_CHAMBER_DEFECT"
