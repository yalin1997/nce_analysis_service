import pandas as pd
import pytest

from nce_analysis.config import AnalysisConfig
from nce_analysis.noise_filter.base import NoiseFilterResult, NoiseFilterStrategy


def test_noise_filter_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        NoiseFilterStrategy()


def test_concrete_subclass_can_implement_filter():
    class DummyStrategy(NoiseFilterStrategy):
        def filter(self, long_df, hotspots, config) -> NoiseFilterResult:
            return NoiseFilterResult(surviving_hotspots=hotspots, litho_self_issues=[])

    strategy = DummyStrategy()
    hotspots = pd.DataFrame({"X_Posi": [0.0], "Y_Posi": [0.0]})
    result = strategy.filter(pd.DataFrame(), hotspots, AnalysisConfig())
    assert result.litho_self_issues == []
    pd.testing.assert_frame_equal(result.surviving_hotspots, hotspots)
