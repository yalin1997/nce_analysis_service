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
    assert config.root_cause_granularity == "chamber"
    assert config.drift_strategy == "regression_cusum"
    assert config.alpha == 0.05
    assert config.summary_top_n == 5


def test_config_rejects_invalid_root_cause_strategy():
    with pytest.raises(ValidationError):
        AnalysisConfig(root_cause_strategy="invalid_choice")


def test_config_rejects_invalid_root_cause_granularity():
    with pytest.raises(ValidationError):
        AnalysisConfig(root_cause_granularity="invalid_choice")


def test_config_rejects_invalid_drift_strategy():
    with pytest.raises(ValidationError):
        AnalysisConfig(drift_strategy="invalid_choice")


@pytest.mark.parametrize(
    "field, value",
    [
        ("hotspot_ratio_threshold", -0.1),
        ("hotspot_ratio_threshold", 1.1),
        ("noise_filter_majority_threshold", -0.1),
        ("noise_filter_majority_threshold", 1.1),
        ("alpha", 0.0),
        ("alpha", 1.0),
        ("alpha", -0.1),
        ("alpha", 1.1),
        ("min_wafer_count", 0),
        ("min_wafer_count", -1),
        ("summary_top_n", 0),
        ("summary_top_n", -1),
    ],
)
def test_config_rejects_out_of_range_values(field, value):
    with pytest.raises(ValidationError):
        AnalysisConfig(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("hotspot_ratio_threshold", 0.0),
        ("hotspot_ratio_threshold", 1.0),
        ("noise_filter_majority_threshold", 0.0),
        ("noise_filter_majority_threshold", 1.0),
        ("alpha", 0.01),
        ("min_wafer_count", 1),
        ("summary_top_n", 1),
    ],
)
def test_config_accepts_boundary_values(field, value):
    config = AnalysisConfig(**{field: value})
    assert getattr(config, field) == value


def test_config_rejects_unknown_field():
    with pytest.raises(ValidationError):
        AnalysisConfig(typo_field=1)


def test_from_yaml_overrides_only_specified_fields(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("summary_top_n: 3\nspec_threshold: 20.0\n")

    config = AnalysisConfig.from_yaml(yaml_path)

    assert config.summary_top_n == 3
    assert config.spec_threshold == 20.0
    assert config.min_wafer_count == 5  # untouched field keeps its default


def test_from_yaml_empty_file_uses_all_defaults(tmp_path):
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("")

    config = AnalysisConfig.from_yaml(yaml_path)

    assert config == AnalysisConfig()


def test_from_yaml_missing_file_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        AnalysisConfig.from_yaml(missing_path)


def test_from_yaml_unknown_field_raises_validation_error(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("typo_field: 1\n")

    with pytest.raises(ValidationError):
        AnalysisConfig.from_yaml(yaml_path)
