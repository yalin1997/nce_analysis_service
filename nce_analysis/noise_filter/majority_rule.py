import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.noise_filter.base import NoiseFilterResult, NoiseFilterStrategy
from nce_analysis.schema import RootCauseDetail


class MajorityRule(NoiseFilterStrategy):
    def filter(
        self, long_df: pd.DataFrame, hotspots: pd.DataFrame, config: AnalysisConfig
    ) -> NoiseFilterResult:
        litho_points = long_df.drop_duplicates(subset=["WaferID", "X_Posi", "Y_Posi"]).copy()
        litho_points["is_anomaly"] = litho_points["NCE_Value"] > config.spec_threshold

        surviving_rows = []
        self_issues: list[RootCauseDetail] = []

        for _, hotspot in hotspots.iterrows():
            coord_mask = (
                (litho_points["X_Posi"] == hotspot["X_Posi"])
                & (litho_points["Y_Posi"] == hotspot["Y_Posi"])
                & litho_points["is_anomaly"]
            )
            anomalous = litho_points[coord_mask]
            total_anomalous = len(anomalous)
            if total_anomalous == 0:
                surviving_rows.append(hotspot)
                continue

            combo_counts = pd.Series(
                list(zip(anomalous["ToolID"], anomalous["ChuckID"]))
            ).value_counts()
            (owning_tool, top_chuck), chuck_n = combo_counts.index[0], combo_counts.iloc[0]
            chuck_share = chuck_n / total_anomalous

            if chuck_share > config.noise_filter_majority_threshold:
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
                        Confidence_Score=chuck_share * 100,
                        Root_Cause_Type=root_cause_type,
                        Affected_Coordinates=[(hotspot["X_Posi"], hotspot["Y_Posi"])],
                        Metrics={
                            "chuck_share": chuck_share,
                            "stage_diversity": float(stage_diversity),
                        },
                    )
                )
                continue

            tool_counts = anomalous["ToolID"].value_counts()
            top_tool, tool_n = tool_counts.index[0], tool_counts.iloc[0]
            tool_share = tool_n / total_anomalous

            if tool_share > config.noise_filter_majority_threshold:
                self_issues.append(
                    RootCauseDetail(
                        Suspect_Pre_ToolID=top_tool,
                        Suspect_Pre_ChamberID="N/A",
                        Suspect_Pre_StepID="N/A",
                        Confidence_Score=tool_share * 100,
                        Root_Cause_Type="LITHO_TOOL_ISSUE",
                        Affected_Coordinates=[(hotspot["X_Posi"], hotspot["Y_Posi"])],
                        Metrics={"tool_share": tool_share},
                    )
                )
                continue

            surviving_rows.append(hotspot)

        surviving_df = (
            pd.DataFrame(surviving_rows) if surviving_rows else hotspots.iloc[0:0].copy()
        )
        return NoiseFilterResult(surviving_hotspots=surviving_df, litho_self_issues=self_issues)
