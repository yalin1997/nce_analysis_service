import pandas as pd

from analysis_core import RiskUpliftRanker, UpliftConfig

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import (
    RootCauseCandidate,
    RootCauseStrategy,
    build_suspect_key,
    split_suspect_key,
)


class MLStrategy(RootCauseStrategy):
    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        working = group_df.copy()
        working["suspect_key"] = build_suspect_key(working, config)

        ranked = RiskUpliftRanker(
            UpliftConfig(group_col="suspect_key", event_col="is_anomaly")
        ).rank(working)
        if not ranked:
            return []

        best = ranked[0]
        tool_id, chamber_id = split_suspect_key(best.group, config)
        return [
            RootCauseCandidate(
                suspect_tool_id=tool_id,
                suspect_chamber_id=chamber_id,
                confidence_score=float(best.normalized_share * 100.0),
                metrics={
                    "risk_uplift": best.uplift,
                    "suspect_anomaly_rate": best.metrics["group_event_rate"],
                    "overall_anomaly_rate": best.metrics["overall_event_rate"],
                    "suspect_anomaly_count": best.metrics["group_event_count"],
                    "suspect_sample_size": best.metrics["group_size"],
                    "sample_size": best.metrics["sample_size"],
                    "ml_scoring_method": 1.0,
                },
            )
        ]
