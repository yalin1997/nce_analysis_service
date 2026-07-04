import logging

import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.correlation import Correlation
from nce_analysis.drift.regression_cusum import RegressionCusum
from nce_analysis.hotspot.ratio_threshold import RatioThreshold
from nce_analysis.noise_filter.majority_rule import MajorityRule
from nce_analysis.preprocessing.wide_history_reshape import WideHistoryReshape
from nce_analysis.result import aggregate_results
from nce_analysis.root_cause.both import BothStrategy
from nce_analysis.root_cause.ml import MLStrategy
from nce_analysis.root_cause.statistical import StatisticalStrategy
from nce_analysis.schema import AnalysisResult, InsufficientSampleHotspot, RootCauseDetail

logger = logging.getLogger(__name__)

_ROOT_CAUSE_STRATEGIES = {
    "statistical": StatisticalStrategy,
    "ml": MLStrategy,
    "both": BothStrategy,
}
_DRIFT_STRATEGIES = {
    "regression_cusum": RegressionCusum,
    "correlation": Correlation,
}


def run(raw_df: pd.DataFrame, config: AnalysisConfig | None = None) -> AnalysisResult:
    config = config or AnalysisConfig()

    preprocessor = WideHistoryReshape()
    hotspot_detector = RatioThreshold()
    noise_filter = MajorityRule()
    root_cause_strategy = _ROOT_CAUSE_STRATEGIES[config.root_cause_strategy]()
    drift_strategy = _DRIFT_STRATEGIES[config.drift_strategy]()

    long_df = preprocessor.transform(raw_df)
    hotspots = hotspot_detector.detect(long_df, config)

    if hotspots.empty:
        return aggregate_results([], config)

    insufficient_hotspots = [
        InsufficientSampleHotspot(
            X_Posi=row["X_Posi"],
            Y_Posi=row["Y_Posi"],
            anomaly_ratio=row["anomaly_ratio"],
            anomalous_wafer_count=int(row["anomalous_wafer_count"]),
            total_wafer_count=int(row["total_wafer_count"]),
        )
        for _, row in hotspots[hotspots["insufficient_sample"]].iterrows()
    ]
    hotspots = hotspots[~hotspots["insufficient_sample"]]

    if hotspots.empty:
        return aggregate_results([], config, insufficient_hotspots)

    noise_result = noise_filter.filter(long_df, hotspots, config)
    details: list[RootCauseDetail] = list(noise_result.litho_self_issues)

    for _, hotspot in noise_result.surviving_hotspots.iterrows():
        coord = (hotspot["X_Posi"], hotspot["Y_Posi"])
        point_rows = long_df[
            (long_df["X_Posi"] == coord[0]) & (long_df["Y_Posi"] == coord[1])
        ]

        # fillna keeps rows from a malformed history level (StageID present but
        # no StepID column) visible instead of silently vanishing from groupby.
        point_rows = point_rows.assign(
            Pre_StepID=point_rows["Pre_StepID"].fillna("UNKNOWN")
        )

        for (pre_stage_id, pre_step_id), step_group in point_rows.groupby(
            ["Pre_StageID", "Pre_StepID"]
        ):
            litho_points = (
                step_group.sort_values("Pre_Execute_Time")
                .drop_duplicates(subset=["WaferID"], keep="last")
                .copy()
            )
            if len(litho_points) < len(step_group):
                logger.warning(
                    "Duplicate wafer records for stage %s step %s at %s "
                    "(rework?); keeping the most recent Pre_Execute_Time per wafer.",
                    pre_stage_id,
                    pre_step_id,
                    coord,
                )
            litho_points["is_anomaly"] = (
                litho_points["NCE_Value"] > config.spec_threshold
            )

            candidates = root_cause_strategy.analyze(litho_points, config)

            for candidate in candidates:
                combo_history = litho_points[
                    (litho_points["Pre_ToolID"] == candidate.suspect_tool_id)
                    & (litho_points["Pre_ChamberID"] == candidate.suspect_chamber_id)
                ]

                if len(combo_history) < 3:
                    root_cause_type = "SPECIFIC_CHAMBER_DEFECT"
                    drift_metrics = {"insufficient_drift_sample": 1.0}
                else:
                    root_cause_type, drift_metrics = drift_strategy.classify(
                        combo_history[["Pre_Execute_Time", "NCE_Value"]], config
                    )

                details.append(
                    RootCauseDetail(
                        Suspect_Pre_ToolID=candidate.suspect_tool_id,
                        Suspect_Pre_ChamberID=candidate.suspect_chamber_id,
                        Suspect_Pre_StepID=pre_step_id,
                        Confidence_Score=candidate.confidence_score,
                        Root_Cause_Type=root_cause_type,
                        Affected_Coordinates=[coord],
                        Metrics={**candidate.metrics, **drift_metrics},
                        Requires_Manual_Review=candidate.requires_manual_review,
                    )
                )

    return aggregate_results(details, config, insufficient_hotspots)
