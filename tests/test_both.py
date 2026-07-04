from unittest.mock import MagicMock

import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import RootCauseCandidate
from nce_analysis.root_cause.both import BothStrategy


def test_both_returns_single_merged_candidate_when_strategies_agree():
    strategy = BothStrategy()
    strategy._statistical = MagicMock()
    strategy._ml = MagicMock()
    strategy._statistical.analyze.return_value = [
        RootCauseCandidate("CMP_01", "ChamberA", 80.0, {"p_value_combo": 0.01})
    ]
    strategy._ml.analyze.return_value = [
        RootCauseCandidate("CMP_01", "ChamberA", 70.0, {"shap_contribution": 0.3})
    ]

    result = strategy.analyze(pd.DataFrame(), AnalysisConfig())

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_01"
    assert result[0].confidence_score == 80.0
    assert result[0].requires_manual_review is False


def test_both_returns_both_candidates_when_strategies_disagree():
    strategy = BothStrategy()
    strategy._statistical = MagicMock()
    strategy._ml = MagicMock()
    strategy._statistical.analyze.return_value = [
        RootCauseCandidate("CMP_01", "ChamberA", 80.0, {})
    ]
    strategy._ml.analyze.return_value = [
        RootCauseCandidate("CMP_02", "ChamberC", 70.0, {})
    ]

    result = strategy.analyze(pd.DataFrame(), AnalysisConfig())

    assert len(result) == 2
    assert all(c.requires_manual_review for c in result)


def test_both_returns_empty_when_neither_strategy_finds_a_suspect():
    strategy = BothStrategy()
    strategy._statistical = MagicMock()
    strategy._ml = MagicMock()
    strategy._statistical.analyze.return_value = []
    strategy._ml.analyze.return_value = []

    result = strategy.analyze(pd.DataFrame(), AnalysisConfig())

    assert result == []
