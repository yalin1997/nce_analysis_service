import pandas as pd
import pytest

from analysis_core.association import AssociationConfig, ChiSquareFisherRanker


def make_df(rows: list[tuple[str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["group", "is_event"])


def test_flags_strongly_associated_group():
    rows = (
        [("bad", True)] * 18 + [("bad", False)] * 2
        + [("good", True)] * 2 + [("good", False)] * 18
    )
    ranked = ChiSquareFisherRanker(AssociationConfig()).rank(make_df(rows))
    assert ranked
    best = ranked[0]
    assert best.group == "bad"
    assert best.odds_ratio > 1
    assert best.p_value_adjusted < 0.05
    assert best.metrics["fisher_fallback"] == 0.0
    assert best.metrics["num_hypotheses"] == 2.0
    assert best.metrics["sample_size"] == 40.0
    assert "p_value_global" in best.metrics


def test_single_group_returns_empty():
    rows = [("only", True)] * 5 + [("only", False)] * 5
    assert ChiSquareFisherRanker(AssociationConfig()).rank(make_df(rows)) == []


def test_no_association_returns_empty():
    rows = (
        [("a", True)] * 10 + [("a", False)] * 10
        + [("b", True)] * 10 + [("b", False)] * 10
    )
    assert ChiSquareFisherRanker(AssociationConfig()).rank(make_df(rows)) == []


def test_small_expected_counts_trigger_fisher_fallback():
    rows = [("bad", True)] * 6 + [("good", True)] * 1 + [("good", False)] * 6
    ranked = ChiSquareFisherRanker(AssociationConfig()).rank(make_df(rows))
    assert ranked
    assert ranked[0].group == "bad"
    assert ranked[0].metrics["fisher_fallback"] == 1.0
    assert "p_value_global" not in ranked[0].metrics


def test_custom_column_names():
    df = pd.DataFrame(
        {
            "chamber": ["x"] * 20 + ["y"] * 20,
            "fail": [True] * 18 + [False] * 2 + [True] * 2 + [False] * 18,
        }
    )
    config = AssociationConfig(group_col="chamber", event_col="fail")
    ranked = ChiSquareFisherRanker(config).rank(df)
    assert ranked and ranked[0].group == "x"


def test_missing_column_raises():
    with pytest.raises(ValueError, match="is_event"):
        ChiSquareFisherRanker(AssociationConfig()).rank(pd.DataFrame({"group": ["a"]}))
