import numpy as np
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
    # n=20 is too small a sample for the sample-size-aware threshold
    # (3 * residual_std * sqrt(n)) to clear on a clean step function: for a
    # noise-free step, cusum_range grows ~O(n) while the threshold grows
    # ~O(sqrt(n)), so the ratio between them is independent of the step's
    # magnitude and only crosses 1 once n is large enough. n=150 gives a
    # comfortable margin.
    n = 150
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


def test_classify_handles_constant_timestamps_without_nan_metrics():
    n = 5
    same_time = pd.Timestamp("2026-01-01")
    series_df = pd.DataFrame(
        {"Pre_Execute_Time": [same_time] * n, "NCE_Value": [10.0, 12.0, 11.0, 13.0, 9.0]}
    )
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, metrics = RegressionCusum().classify(series_df, config)

    assert root_cause_type == "SPECIFIC_CHAMBER_DEFECT"
    assert metrics["insufficient_time_variation"] == 1.0
    assert not any(value != value for value in metrics.values())


def test_classify_does_not_report_sudden_shift_for_seeded_white_noise():
    # Seed 1 is used (rather than 0) because with n=50 the linear-regression
    # slope test on seed 0's noise happens to land at p=0.0073 by chance,
    # which would trip the unrelated CHAMBER_DRIFT path and confound this
    # CUSUM-specific regression test.
    rng = np.random.default_rng(1)
    n = 50
    values = 20.0 + rng.normal(0.0, 1.0, n)
    series_df = pd.DataFrame(
        {"Pre_Execute_Time": _timestamps(n), "NCE_Value": values}
    )
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, metrics = RegressionCusum().classify(series_df, config)

    assert root_cause_type == "SPECIFIC_CHAMBER_DEFECT"
    assert metrics["cusum_threshold"] > metrics["cusum_range"]
    assert metrics["cusum_threshold_method"] == 1.0
