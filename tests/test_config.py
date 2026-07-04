import pytest
from pydantic import ValidationError

from nce_analysis.config import AnalysisConfig


def test_default_config_values():
    config = AnalysisConfig()
    assert config.spec_threshold == 15.0
    assert config.min_wafer_count == 5
    assert config.hotspot_ratio_threshold == 0.05
    assert config.noise_filter_majority_threshold == 0.5
    assert config.root_cause_strategy == "both"
    assert config.drift_strategy == "regression_cusum"
    assert config.alpha == 0.05
    assert config.summary_top_n == 5


def test_config_rejects_invalid_root_cause_strategy():
    with pytest.raises(ValidationError):
        AnalysisConfig(root_cause_strategy="invalid_choice")


def test_config_rejects_invalid_drift_strategy():
    with pytest.raises(ValidationError):
        AnalysisConfig(drift_strategy="invalid_choice")
