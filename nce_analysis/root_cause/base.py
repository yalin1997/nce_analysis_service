from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.schema import RootCauseDetail

SUSPECT_KEY_SEPARATOR = "|"
_LITHO_SELF_TYPES = ("LITHO_CHUCK_ISSUE", "LITHO_CHUCK_CONTAMINATION", "LITHO_TOOL_ISSUE")


@dataclass
class RootCauseCandidate:
    suspect_tool_id: str
    suspect_chamber_id: str
    confidence_score: float
    metrics: dict[str, float] = field(default_factory=dict)
    requires_manual_review: bool = False


def build_suspect_key(df: pd.DataFrame, config: AnalysisConfig) -> pd.Series:
    """Suspect-key column strategies group/classify by: Pre_ToolID alone when
    config.root_cause_granularity == "tool", else Pre_ToolID + Pre_ChamberID
    combined. Inverse of split_suspect_key."""
    if config.root_cause_granularity == "tool":
        return df["Pre_ToolID"]
    return df["Pre_ToolID"] + SUSPECT_KEY_SEPARATOR + df["Pre_ChamberID"]


def split_suspect_key(suspect_key: str, config: AnalysisConfig) -> tuple[str, str]:
    """Inverse of build_suspect_key: recover (tool_id, chamber_id). chamber_id
    is "N/A" when granularity is "tool" (no chamber dimension was used)."""
    if config.root_cause_granularity == "tool":
        return suspect_key, "N/A"
    tool_id, chamber_id = suspect_key.split(SUSPECT_KEY_SEPARATOR, 1)
    return tool_id, chamber_id


def matches_suspect(
    df: pd.DataFrame, candidate: "RootCauseCandidate", config: AnalysisConfig
) -> pd.Series:
    """Boolean mask selecting df rows belonging to candidate's suspect group,
    e.g. to pull a suspect's full history for drift analysis. Respects
    config.root_cause_granularity via build_suspect_key rather than
    re-deriving the tool-vs-tool+chamber decision at the call site."""
    if config.root_cause_granularity == "tool":
        candidate_key = candidate.suspect_tool_id
    else:
        candidate_key = candidate.suspect_tool_id + SUSPECT_KEY_SEPARATOR + candidate.suspect_chamber_id
    return build_suspect_key(df, config) == candidate_key


def _coord_mask(df: pd.DataFrame, coordinates: list[tuple[float, float]]) -> pd.Series:
    coord_set = set(coordinates)
    return pd.Series(
        list(zip(df["X_Posi"], df["Y_Posi"])), index=df.index
    ).isin(coord_set)


def _resolve_litho_self_history(
    long_df: pd.DataFrame, detail: RootCauseDetail, config: AnalysisConfig, coord_mask: pd.Series
) -> pd.DataFrame:
    if detail.Root_Cause_Type == "LITHO_TOOL_ISSUE":
        own_mask = long_df["ToolID"] == detail.Suspect_Pre_ToolID
    else:
        own_mask = (long_df["ToolID"] == detail.Suspect_Pre_ToolID) & (
            long_df["ChuckID"] == detail.Suspect_Pre_ChamberID
        )
    stage_ids = set(long_df.loc[own_mask & coord_mask, "StageID"].unique())

    scope = long_df[long_df["StageID"].isin(stage_ids) & coord_mask].copy()
    scope = (
        scope.sort_values("Execute_Time")
        .drop_duplicates(subset=["WaferID", "X_Posi", "Y_Posi"], keep="last")
    )
    scope["is_anomaly"] = scope["NCE_Value"] > config.spec_threshold

    if detail.Root_Cause_Type == "LITHO_TOOL_ISSUE":
        scope["is_suspect_group"] = scope["ToolID"] == detail.Suspect_Pre_ToolID
    else:
        scope["is_suspect_group"] = (scope["ToolID"] == detail.Suspect_Pre_ToolID) & (
            scope["ChuckID"] == detail.Suspect_Pre_ChamberID
        )

    group_label = (scope["ToolID"] + SUSPECT_KEY_SEPARATOR + scope["ChuckID"]).fillna("UNKNOWN")
    return pd.DataFrame(
        {
            "time": scope["Execute_Time"],
            "NCE_Value": scope["NCE_Value"],
            "group_label": group_label,
            "WaferID": scope["WaferID"],
            "is_anomaly": scope["is_anomaly"],
            "is_suspect_group": scope["is_suspect_group"],
        }
    )


def _resolve_upstream_history(
    long_df: pd.DataFrame, detail: RootCauseDetail, config: AnalysisConfig, coord_mask: pd.Series
) -> pd.DataFrame:
    # fillna mirrors pipeline.py's NaN->"UNKNOWN" Pre_StepID substitution so a
    # detail keyed on the "UNKNOWN" sentinel still matches raw NaN rows here
    # (this long_df may come straight from WideHistoryReshape, unlike
    # pipeline.py's already-filled point_rows).
    step_ids = long_df["Pre_StepID"].fillna("UNKNOWN")
    step_mask = step_ids == detail.Suspect_Pre_StepID
    scope = (
        long_df[step_mask & coord_mask]
        .sort_values("Pre_Execute_Time")
        .drop_duplicates(subset=["WaferID", "X_Posi", "Y_Posi"], keep="last")
        .copy()
    )
    scope["is_anomaly"] = scope["NCE_Value"] > config.spec_threshold

    candidate = RootCauseCandidate(
        suspect_tool_id=detail.Suspect_Pre_ToolID,
        suspect_chamber_id=detail.Suspect_Pre_ChamberID,
        confidence_score=0.0,
    )
    scope["is_suspect_group"] = matches_suspect(scope, candidate, config)
    group_label = build_suspect_key(scope, config).fillna("UNKNOWN")

    return pd.DataFrame(
        {
            "time": scope["Pre_Execute_Time"],
            "NCE_Value": scope["NCE_Value"],
            "group_label": group_label,
            "WaferID": scope["WaferID"],
            "is_anomaly": scope["is_anomaly"],
            "is_suspect_group": scope["is_suspect_group"],
        }
    )


def resolve_detail_history(
    long_df: pd.DataFrame, detail: RootCauseDetail, config: AnalysisConfig
) -> pd.DataFrame:
    """Given one RootCauseDetail from AnalysisResult, reconstruct the
    wafer-level rows it was computed from. Returns columns normalized for
    chart consumption: time, NCE_Value, group_label, WaferID, is_anomaly,
    is_suspect_group."""
    coord_mask = _coord_mask(long_df, detail.Affected_Coordinates)
    if detail.Root_Cause_Type in _LITHO_SELF_TYPES:
        return _resolve_litho_self_history(long_df, detail, config, coord_mask)
    return _resolve_upstream_history(long_df, detail, config, coord_mask)


class RootCauseStrategy(ABC):
    @abstractmethod
    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        """group_df: rows for wafers at one hotspot coordinate and a single
        (Pre_StageID, Pre_StepID) group, with columns Pre_ToolID,
        Pre_ChamberID, is_anomaly.
        Returns a list of suspect Pre_ToolID+Pre_ChamberID candidates (usually
        0 or 1 entries; the 'both' cross-validation strategy may return 2 when
        its sub-strategies disagree, each marked requires_manual_review=True)."""
