import pandas as pd
import pytest

from analysis_core.uplift import GroupUplift, RiskUpliftRanker, UpliftConfig


def make_df(rows: list[tuple[str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["group", "is_event"])


def test_ranks_positive_uplift_groups_descending():
    rows = (
        [("worst", True)] * 8 + [("worst", False)] * 2
        + [("mid", True)] * 5 + [("mid", False)] * 5
        + [("clean", False)] * 10
    )
    ranked = RiskUpliftRanker(UpliftConfig()).rank(make_df(rows))
    assert [item.group for item in ranked] == ["worst", "mid"]
    assert ranked[0].uplift > ranked[1].uplift > 0
    assert abs(sum(item.normalized_share for item in ranked) - 1.0) < 1e-9
    assert ranked[0].metrics["group_event_rate"] == 0.8
    assert ranked[0].metrics["group_size"] == 10.0
    assert ranked[0].metrics["sample_size"] == 30.0


def test_single_group_returns_empty():
    rows = [("only", True), ("only", False)]
    assert RiskUpliftRanker(UpliftConfig()).rank(make_df(rows)) == []


def test_all_events_returns_empty():
    rows = [("a", True)] * 3 + [("b", True)] * 3
    assert RiskUpliftRanker(UpliftConfig()).rank(make_df(rows)) == []


def test_no_events_returns_empty():
    rows = [("a", False)] * 3 + [("b", False)] * 3
    assert RiskUpliftRanker(UpliftConfig()).rank(make_df(rows)) == []


def test_custom_column_names():
    df = pd.DataFrame(
        {
            "chamber": ["x"] * 5 + ["y"] * 5,
            "fail": [True] * 4 + [False] * 1 + [False] * 5,
        }
    )
    config = UpliftConfig(group_col="chamber", event_col="fail")
    ranked = RiskUpliftRanker(config).rank(df)
    assert ranked and ranked[0].group == "x"


def test_missing_column_raises():
    with pytest.raises(ValueError, match="is_event"):
        RiskUpliftRanker(UpliftConfig()).rank(pd.DataFrame({"group": ["a"]}))
