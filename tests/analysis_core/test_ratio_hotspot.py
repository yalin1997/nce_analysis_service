import pandas as pd
import pytest

from analysis_core.ratio_hotspot import RatioHotspotConfig, RatioHotspotDetector


def test_groups_filtered_by_ratio_and_sorted_descending():
    df = pd.DataFrame(
        {
            "x": [0] * 10 + [1] * 10 + [2] * 10,
            "y": [0] * 30,
            "is_event": (
                [True] * 8 + [False] * 2      # (0,0): ratio 0.8
                + [True] * 1 + [False] * 9    # (1,0): ratio 0.1
                + [False] * 10                # (2,0): ratio 0.0 -> filtered
            ),
        }
    )
    config = RatioHotspotConfig(
        group_cols=["x", "y"], ratio_threshold=0.05, min_count=5
    )
    detected = RatioHotspotDetector(config).detect(df)
    assert list(detected.columns) == [
        "x", "y", "event_count", "total_count", "event_ratio", "insufficient_sample",
    ]
    assert list(detected["event_ratio"]) == [0.8, 0.1]
    assert list(detected["event_count"]) == [8, 1]
    assert not detected["insufficient_sample"].any()


def test_small_groups_kept_but_flagged_insufficient():
    df = pd.DataFrame(
        {
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "is_event": [True, True, False],
        }
    )
    config = RatioHotspotConfig(
        group_cols=["x", "y"], ratio_threshold=0.05, min_count=5
    )
    detected = RatioHotspotDetector(config).detect(df)
    assert len(detected) == 1
    assert bool(detected.loc[0, "insufficient_sample"]) is True


def test_missing_column_raises():
    config = RatioHotspotConfig(group_cols=["x"], event_col="is_event")
    with pytest.raises(ValueError, match="is_event"):
        RatioHotspotDetector(config).detect(pd.DataFrame({"x": [1]}))
