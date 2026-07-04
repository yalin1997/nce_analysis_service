import pandas as pd
import pytest

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.base import DriftStrategy


def test_drift_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        DriftStrategy()


def test_concrete_subclass_can_implement_classify():
    class DummyStrategy(DriftStrategy):
        def classify(self, series_df: pd.DataFrame, config: AnalysisConfig):
            return "SPECIFIC_CHAMBER_DEFECT", {}

    strategy = DummyStrategy()
    result = strategy.classify(pd.DataFrame(), AnalysisConfig())
    assert result == ("SPECIFIC_CHAMBER_DEFECT", {})
