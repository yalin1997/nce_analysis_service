import pandas as pd
import pytest

from nce_analysis.config import AnalysisConfig
from nce_analysis.hotspot.base import HotspotStrategy


def test_hotspot_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        HotspotStrategy()


def test_concrete_subclass_can_implement_detect():
    class DummyStrategy(HotspotStrategy):
        def detect(self, long_df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
            return long_df

    strategy = DummyStrategy()
    df = pd.DataFrame({"a": [1, 2]})
    result = strategy.detect(df, AnalysisConfig())
    pd.testing.assert_frame_equal(result, df)
