import pandas as pd
import pytest

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import RootCauseCandidate, RootCauseStrategy


def test_root_cause_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        RootCauseStrategy()


def test_root_cause_candidate_defaults():
    candidate = RootCauseCandidate(
        suspect_tool_id="CMP_01", suspect_chamber_id="ChamberA", confidence_score=90.0
    )
    assert candidate.metrics == {}
    assert candidate.requires_manual_review is False


def test_concrete_subclass_can_implement_analyze():
    class DummyStrategy(RootCauseStrategy):
        def analyze(self, group_df: pd.DataFrame, config: AnalysisConfig):
            return []

    strategy = DummyStrategy()
    result = strategy.analyze(pd.DataFrame(), AnalysisConfig())
    assert result == []
