import pandas as pd
import pytest

from nce_analysis.preprocessing.base import PreprocessingStrategy


def test_preprocessing_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        PreprocessingStrategy()


def test_concrete_subclass_can_implement_transform():
    class DummyStrategy(PreprocessingStrategy):
        def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
            return raw_df

    strategy = DummyStrategy()
    df = pd.DataFrame({"a": [1, 2]})
    result = strategy.transform(df)
    pd.testing.assert_frame_equal(result, df)
