from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from nce_analysis.config import AnalysisConfig
from nce_analysis.schema import AnalysisResult, RootCauseDetail


def _detail(**overrides):
    defaults = dict(
        Suspect_Pre_ToolID="CMP_01",
        Suspect_Pre_ChamberID="ChamberA",
        Suspect_Pre_StepID="3580.01",
        Confidence_Score=92.5,
        Root_Cause_Type="SPECIFIC_CHAMBER_DEFECT",
        Affected_Coordinates=[(0.0, 0.0)],
        Metrics={"p_value": 0.001, "odds_ratio": 8.2},
    )
    defaults.update(overrides)
    return RootCauseDetail(**defaults)


def test_root_cause_detail_defaults_requires_manual_review_false():
    detail = _detail()
    assert detail.Requires_Manual_Review is False


def test_root_cause_detail_rejects_invalid_root_cause_type():
    with pytest.raises(ValidationError):
        _detail(Root_Cause_Type="NOT_A_REAL_TYPE")


def test_analysis_result_construction():
    detail = _detail()
    result = AnalysisResult(
        Summary=[detail],
        Details=[detail],
        Generated_At=datetime.now(timezone.utc),
        Config_Used=AnalysisConfig(),
    )
    assert result.Summary[0].Suspect_Pre_ToolID == "CMP_01"
    assert result.Config_Used.spec_threshold == 15.0
    assert result.Insufficient_Sample_Hotspots == []
