import pandas as pd

from analysis_core import DominanceConfig, find_dominant

from nce_analysis.config import AnalysisConfig
from nce_analysis.noise_filter.base import NoiseFilterResult, NoiseFilterStrategy
from nce_analysis.schema import RootCauseDetail


class MajorityRule(NoiseFilterStrategy):
    def filter(
        self, long_df: pd.DataFrame, hotspots: pd.DataFrame, config: AnalysisConfig
    ) -> NoiseFilterResult:
        litho_points = long_df.drop_duplicates(subset=["WaferID", "X_Posi", "Y_Posi"]).copy()
        litho_points["is_anomaly"] = litho_points["NCE_Value"] > config.spec_threshold

        dominance_config = DominanceConfig(threshold=config.noise_filter_majority_threshold)
        surviving_rows = []
        self_issues: list[RootCauseDetail] = []

        for _, hotspot in hotspots.iterrows():
            coord_mask = (
                (litho_points["X_Posi"] == hotspot["X_Posi"])
                & (litho_points["Y_Posi"] == hotspot["Y_Posi"])
                & litho_points["is_anomaly"]
            )
            anomalous = litho_points[coord_mask]
            if len(anomalous) == 0:
                surviving_rows.append(hotspot)
                continue

            chuck_dominance = find_dominant(
                pd.Series(list(zip(anomalous["ToolID"], anomalous["ChuckID"]))),
                dominance_config,
            )
            if chuck_dominance is not None:
                owning_tool, top_chuck = chuck_dominance.category
                chuck_rows = anomalous[
                    (anomalous["ToolID"] == owning_tool) & (anomalous["ChuckID"] == top_chuck)
                ]
                stage_diversity = chuck_rows["StageID"].nunique()
                root_cause_type = (
                    "LITHO_CHUCK_CONTAMINATION" if stage_diversity > 1 else "LITHO_CHUCK_ISSUE"
                )
                self_issues.append(
                    RootCauseDetail(
                        Suspect_Pre_ToolID=owning_tool,
                        Suspect_Pre_ChamberID=top_chuck,
                        Suspect_Pre_StepID="N/A",
                        Confidence_Score=chuck_dominance.share * 100,
                        Root_Cause_Type=root_cause_type,
                        Affected_Coordinates=[(hotspot["X_Posi"], hotspot["Y_Posi"])],
                        Metrics={
                            "chuck_share": chuck_dominance.share,
                            "stage_diversity": float(stage_diversity),
                        },
                    )
                )
                continue

            tool_dominance = find_dominant(anomalous["ToolID"], dominance_config)
            if tool_dominance is not None:
                (top_tool,) = tool_dominance.category
                self_issues.append(
                    RootCauseDetail(
                        Suspect_Pre_ToolID=top_tool,
                        Suspect_Pre_ChamberID="N/A",
                        Suspect_Pre_StepID="N/A",
                        Confidence_Score=tool_dominance.share * 100,
                        Root_Cause_Type="LITHO_TOOL_ISSUE",
                        Affected_Coordinates=[(hotspot["X_Posi"], hotspot["Y_Posi"])],
                        Metrics={"tool_share": tool_dominance.share},
                    )
                )
                continue

            surviving_rows.append(hotspot)

        surviving_df = (
            pd.DataFrame(surviving_rows) if surviving_rows else hotspots.iloc[0:0].copy()
        )
        return NoiseFilterResult(surviving_hotspots=surviving_df, litho_self_issues=self_issues)
