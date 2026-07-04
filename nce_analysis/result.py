from datetime import datetime, timezone

from nce_analysis.config import AnalysisConfig
from nce_analysis.schema import AnalysisResult, InsufficientSampleHotspot, RootCauseDetail


def aggregate_results(
    details: list[RootCauseDetail],
    config: AnalysisConfig,
    insufficient_sample_hotspots: list[InsufficientSampleHotspot] | None = None,
) -> AnalysisResult:
    sorted_details = sorted(details, key=lambda d: d.Confidence_Score, reverse=True)

    # Summary: one entry per suspect. Iterating in confidence order means the
    # first detail seen per key carries the group's max confidence and metrics,
    # and dict insertion order keeps the summary confidence-sorted.
    merged: dict[tuple, RootCauseDetail] = {}
    for detail in sorted_details:
        key = (
            detail.Suspect_Pre_ToolID,
            detail.Suspect_Pre_ChamberID,
            detail.Suspect_Pre_StepID,
            detail.Root_Cause_Type,
        )
        if key not in merged:
            merged[key] = detail.model_copy(deep=True)
            continue
        entry = merged[key]
        entry.Affected_Coordinates = entry.Affected_Coordinates + [
            coord
            for coord in detail.Affected_Coordinates
            if coord not in entry.Affected_Coordinates
        ]
        entry.Requires_Manual_Review = (
            entry.Requires_Manual_Review or detail.Requires_Manual_Review
        )

    return AnalysisResult(
        Summary=list(merged.values())[: config.summary_top_n],
        Details=sorted_details,
        Insufficient_Sample_Hotspots=insufficient_sample_hotspots or [],
        Generated_At=datetime.now(timezone.utc),
        Config_Used=config,
    )
