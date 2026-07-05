import pandas as pd
import pytest

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import (
    RootCauseCandidate,
    RootCauseStrategy,
    build_suspect_key,
    matches_suspect,
    resolve_detail_history,
    split_suspect_key,
)
from nce_analysis.schema import RootCauseDetail


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


def test_build_suspect_key_combines_tool_and_chamber_by_default():
    df = pd.DataFrame({"Pre_ToolID": ["CMP_01", "CMP_02"], "Pre_ChamberID": ["A", "B"]})

    key = build_suspect_key(df, AnalysisConfig())

    assert key.tolist() == ["CMP_01|A", "CMP_02|B"]


def test_build_suspect_key_uses_tool_only_when_granularity_is_tool():
    df = pd.DataFrame({"Pre_ToolID": ["CMP_01", "CMP_02"], "Pre_ChamberID": ["A", "B"]})

    key = build_suspect_key(df, AnalysisConfig(root_cause_granularity="tool"))

    assert key.tolist() == ["CMP_01", "CMP_02"]


def test_split_suspect_key_round_trips_chamber_granularity():
    tool_id, chamber_id = split_suspect_key("CMP_01|A", AnalysisConfig())

    assert (tool_id, chamber_id) == ("CMP_01", "A")


def test_split_suspect_key_returns_na_chamber_for_tool_granularity():
    tool_id, chamber_id = split_suspect_key(
        "CMP_01", AnalysisConfig(root_cause_granularity="tool")
    )

    assert (tool_id, chamber_id) == ("CMP_01", "N/A")


def _combo_df():
    return pd.DataFrame(
        {
            "Pre_ToolID": ["CMP_01", "CMP_01", "CMP_02"],
            "Pre_ChamberID": ["ChamberA", "ChamberB", "ChamberA"],
        }
    )


def test_matches_suspect_filters_by_tool_only_when_granularity_is_tool():
    candidate = RootCauseCandidate(
        suspect_tool_id="CMP_01", suspect_chamber_id="N/A", confidence_score=90.0
    )
    config = AnalysisConfig(root_cause_granularity="tool")

    mask = matches_suspect(_combo_df(), candidate, config)

    assert mask.tolist() == [True, True, False]


def test_matches_suspect_filters_by_tool_and_chamber_by_default():
    candidate = RootCauseCandidate(
        suspect_tool_id="CMP_01", suspect_chamber_id="ChamberA", confidence_score=90.0
    )
    config = AnalysisConfig()

    mask = matches_suspect(_combo_df(), candidate, config)

    assert mask.tolist() == [True, False, False]


def _upstream_detail(**overrides):
    defaults = dict(
        Suspect_Pre_ToolID="CMP_01",
        Suspect_Pre_ChamberID="ChamberA",
        Suspect_Pre_StepID="STEP1",
        Confidence_Score=90.0,
        Root_Cause_Type="SPECIFIC_CHAMBER_DEFECT",
        Affected_Coordinates=[(0.0, 0.0)],
        Metrics={},
    )
    defaults.update(overrides)
    return RootCauseDetail(**defaults)


def test_resolve_detail_history_upstream_tags_suspect_and_peer_rows():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": "W1",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "Pre_ToolID": "CMP_01",
                "Pre_ChamberID": "ChamberA",
                "Pre_StepID": "STEP1",
                "Pre_Execute_Time": "2025-01-01",
            },
            {
                "WaferID": "W2",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 5.0,
                "Pre_ToolID": "CMP_02",
                "Pre_ChamberID": "ChamberB",
                "Pre_StepID": "STEP1",
                "Pre_Execute_Time": "2025-01-01",
            },
            {
                "WaferID": "W3",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 5.0,
                "Pre_ToolID": "CMP_01",
                "Pre_ChamberID": "ChamberA",
                "Pre_StepID": "STEP2",
                "Pre_Execute_Time": "2025-01-01",
            },
            {
                "WaferID": "W4",
                "X_Posi": 5.0,
                "Y_Posi": 5.0,
                "NCE_Value": 5.0,
                "Pre_ToolID": "CMP_01",
                "Pre_ChamberID": "ChamberA",
                "Pre_StepID": "STEP1",
                "Pre_Execute_Time": "2025-01-01",
            },
        ]
    )
    detail = _upstream_detail()
    config = AnalysisConfig()

    result = resolve_detail_history(long_df, detail, config)

    assert set(result["WaferID"]) == {"W1", "W2"}
    suspect_row = result[result["WaferID"] == "W1"].iloc[0]
    peer_row = result[result["WaferID"] == "W2"].iloc[0]
    assert suspect_row["is_suspect_group"] is True or bool(suspect_row["is_suspect_group"])
    assert not bool(peer_row["is_suspect_group"])
    assert bool(suspect_row["is_anomaly"])
    assert not bool(peer_row["is_anomaly"])
    assert suspect_row["group_label"] == "CMP_01|ChamberA"
    assert peer_row["group_label"] == "CMP_02|ChamberB"
    assert set(result.columns) == {
        "time",
        "NCE_Value",
        "group_label",
        "WaferID",
        "is_anomaly",
        "is_suspect_group",
    }


def test_resolve_detail_history_upstream_dedupes_rework_keeping_latest():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": "W1",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 5.0,
                "Pre_ToolID": "CMP_01",
                "Pre_ChamberID": "ChamberA",
                "Pre_StepID": "STEP1",
                "Pre_Execute_Time": "2025-01-01",
            },
            {
                "WaferID": "W1",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "Pre_ToolID": "CMP_01",
                "Pre_ChamberID": "ChamberA",
                "Pre_StepID": "STEP1",
                "Pre_Execute_Time": "2025-02-01",
            },
        ]
    )
    detail = _upstream_detail()
    config = AnalysisConfig()

    result = resolve_detail_history(long_df, detail, config)

    assert len(result) == 1
    assert result.iloc[0]["NCE_Value"] == 20.0


def test_resolve_detail_history_upstream_missing_chamber_falls_back_to_unknown():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": "W1",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "Pre_ToolID": "CMP_01",
                "Pre_ChamberID": "ChamberA",
                "Pre_StepID": "STEP1",
                "Pre_Execute_Time": "2025-01-01",
            },
            {
                "WaferID": "W2",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 5.0,
                "Pre_ToolID": "CMP_03",
                "Pre_ChamberID": pd.NA,
                "Pre_StepID": "STEP1",
                "Pre_Execute_Time": "2025-01-01",
            },
        ]
    )
    detail = _upstream_detail()
    config = AnalysisConfig()

    result = resolve_detail_history(long_df, detail, config)

    peer_row = result[result["WaferID"] == "W2"].iloc[0]
    assert peer_row["group_label"] == "UNKNOWN"


def _litho_self_detail(**overrides):
    defaults = dict(
        Suspect_Pre_ToolID="LITHO_A",
        Suspect_Pre_ChamberID="CHK_1",
        Suspect_Pre_StepID="N/A",
        Confidence_Score=80.0,
        Root_Cause_Type="LITHO_CHUCK_ISSUE",
        Affected_Coordinates=[(0.0, 0.0)],
        Metrics={},
    )
    defaults.update(overrides)
    return RootCauseDetail(**defaults)


def test_resolve_detail_history_chuck_issue_includes_peers_at_recovered_stage_only():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": "W1",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "StageID": "LITHO_M1",
                "Execute_Time": "2026-01-01",
            },
            {
                "WaferID": "W2",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 5.0,
                "ToolID": "LITHO_B",
                "ChuckID": "CHK_2",
                "StageID": "LITHO_M1",
                "Execute_Time": "2026-01-01",
            },
            {
                "WaferID": "W3",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 5.0,
                "ToolID": "LITHO_C",
                "ChuckID": "CHK_3",
                "StageID": "LITHO_M2",
                "Execute_Time": "2026-01-01",
            },
        ]
    )
    detail = _litho_self_detail(Root_Cause_Type="LITHO_CHUCK_ISSUE")
    config = AnalysisConfig()

    result = resolve_detail_history(long_df, detail, config)

    assert set(result["WaferID"]) == {"W1", "W2"}
    suspect_row = result[result["WaferID"] == "W1"].iloc[0]
    peer_row = result[result["WaferID"] == "W2"].iloc[0]
    assert bool(suspect_row["is_suspect_group"])
    assert not bool(peer_row["is_suspect_group"])
    assert suspect_row["group_label"] == "LITHO_A|CHK_1"
    assert peer_row["group_label"] == "LITHO_B|CHK_2"


def test_resolve_detail_history_tool_issue_tags_by_tool_only():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": "W1",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "StageID": "LITHO_M1",
                "Execute_Time": "2026-01-01",
            },
            {
                "WaferID": "W2",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_2",
                "StageID": "LITHO_M1",
                "Execute_Time": "2026-01-01",
            },
            {
                "WaferID": "W3",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 5.0,
                "ToolID": "LITHO_B",
                "ChuckID": "CHK_3",
                "StageID": "LITHO_M1",
                "Execute_Time": "2026-01-01",
            },
        ]
    )
    detail = _litho_self_detail(
        Root_Cause_Type="LITHO_TOOL_ISSUE", Suspect_Pre_ChamberID="N/A"
    )
    config = AnalysisConfig()

    result = resolve_detail_history(long_df, detail, config)

    assert set(result["WaferID"]) == {"W1", "W2", "W3"}
    assert bool(result[result["WaferID"] == "W1"].iloc[0]["is_suspect_group"])
    assert bool(result[result["WaferID"] == "W2"].iloc[0]["is_suspect_group"])
    assert not bool(result[result["WaferID"] == "W3"].iloc[0]["is_suspect_group"])


def test_resolve_detail_history_chuck_contamination_recovers_both_stages():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": "W1",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "StageID": "LITHO_M1",
                "Execute_Time": "2026-01-01",
            },
            {
                "WaferID": "W2",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "StageID": "LITHO_M2",
                "Execute_Time": "2026-01-01",
            },
            {
                "WaferID": "W3",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 5.0,
                "ToolID": "LITHO_B",
                "ChuckID": "CHK_2",
                "StageID": "LITHO_M1",
                "Execute_Time": "2026-01-01",
            },
            {
                "WaferID": "W4",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 5.0,
                "ToolID": "LITHO_C",
                "ChuckID": "CHK_3",
                "StageID": "LITHO_M2",
                "Execute_Time": "2026-01-01",
            },
            {
                "WaferID": "W5",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 5.0,
                "ToolID": "LITHO_D",
                "ChuckID": "CHK_4",
                "StageID": "LITHO_M3",
                "Execute_Time": "2026-01-01",
            },
        ]
    )
    detail = _litho_self_detail(Root_Cause_Type="LITHO_CHUCK_CONTAMINATION")
    config = AnalysisConfig()

    result = resolve_detail_history(long_df, detail, config)

    assert set(result["WaferID"]) == {"W1", "W2", "W3", "W4"}


def test_resolve_detail_history_self_branch_dedupes_by_wafer_coordinate():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": "W1",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "StageID": "LITHO_M1",
                "Execute_Time": "2026-01-01",
            },
            {
                "WaferID": "W1",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "StageID": "LITHO_M1",
                "Execute_Time": "2026-01-01",
            },
        ]
    )
    detail = _litho_self_detail(Root_Cause_Type="LITHO_CHUCK_ISSUE")
    config = AnalysisConfig()

    result = resolve_detail_history(long_df, detail, config)

    assert len(result) == 1
