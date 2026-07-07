import numpy as np
import pandas as pd

from analysis_core.trend import (
    CorrelationTrendDetector,
    RegressionCusumDetector,
    TrendConfig,
)


def make_series(values) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=len(values), freq="h")
    return pd.DataFrame({"timestamp": times, "value": list(values)})


def test_noisy_increase_is_increasing_trend():
    rng = np.random.default_rng(42)
    values = np.arange(30) * 2.0 + rng.normal(0, 0.5, 30)
    result = RegressionCusumDetector(TrendConfig()).detect(make_series(values))
    assert result.kind == "increasing_trend"
    assert result.metrics["slope"] > 0
    assert result.metrics["slope_p_value"] < 0.05
    assert result.metrics["cusum_threshold_method"] == 1.0


def test_step_jump_is_change_point():
    rng = np.random.default_rng(7)
    values = np.concatenate(
        [rng.normal(10, 0.3, 60), rng.normal(20, 0.3, 60)]
    )
    result = RegressionCusumDetector(TrendConfig()).detect(make_series(values))
    assert result.kind == "change_point"
    assert result.metrics["cusum_range"] > result.metrics["cusum_threshold"]


def test_flat_noise_is_none():
    rng = np.random.default_rng(3)
    values = rng.normal(10, 0.5, 20)
    result = RegressionCusumDetector(TrendConfig()).detect(make_series(values))
    assert result.kind == "none"


def test_constant_timestamps_report_insufficient_time_variation():
    df = pd.DataFrame(
        {"timestamp": ["2026-01-01"] * 5, "value": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    result = RegressionCusumDetector(TrendConfig()).detect(df)
    assert result.kind == "none"
    assert result.metrics == {"insufficient_time_variation": 1.0}


def test_correlation_detector_flags_increasing_trend():
    values = [float(i) for i in range(12)]
    result = CorrelationTrendDetector(TrendConfig()).detect(make_series(values))
    assert result.kind == "increasing_trend"
    assert result.metrics["correlation"] > 0


def test_correlation_constant_values_report_insufficient_value_variation():
    result = CorrelationTrendDetector(TrendConfig()).detect(make_series([5.0] * 6))
    assert result.kind == "none"
    assert result.metrics == {"insufficient_value_variation": 1.0}


def test_custom_column_names():
    df = pd.DataFrame(
        {
            "ran_at": pd.date_range("2026-01-01", periods=12, freq="h"),
            "reading": [float(i) for i in range(12)],
        }
    )
    config = TrendConfig(time_col="ran_at", value_col="reading")
    result = CorrelationTrendDetector(config).detect(df)
    assert result.kind == "increasing_trend"
