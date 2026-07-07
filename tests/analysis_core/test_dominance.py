import pandas as pd

from analysis_core.dominance import Dominance, DominanceConfig, find_dominant


def test_majority_category_found():
    keys = pd.Series(["a"] * 7 + ["b"] * 3)
    assert find_dominant(keys, DominanceConfig(threshold=0.5)) == Dominance(
        category=("a",), share=0.7, count=7
    )


def test_share_equal_to_threshold_is_not_dominant():
    keys = pd.Series(["a"] * 5 + ["b"] * 5)
    assert find_dominant(keys, DominanceConfig(threshold=0.5)) is None


def test_composite_keys_returned_as_is():
    keys = pd.Series([("t1", "c1")] * 3 + [("t2", "c2")] * 1)
    dominance = find_dominant(keys, DominanceConfig(threshold=0.5))
    assert dominance is not None
    assert dominance.category == ("t1", "c1")
    assert dominance.count == 3


def test_empty_series_returns_none():
    keys = pd.Series([], dtype=object)
    assert find_dominant(keys, DominanceConfig()) is None
