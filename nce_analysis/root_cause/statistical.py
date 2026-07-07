import pandas as pd

from analysis_core import AssociationConfig, ChiSquareFisherRanker

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import (
    RootCauseCandidate,
    RootCauseStrategy,
    build_suspect_key,
    split_suspect_key,
)


class StatisticalStrategy(RootCauseStrategy):
    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        working = group_df.copy()
        working["suspect_key"] = build_suspect_key(working, config)

        ranked = ChiSquareFisherRanker(
            AssociationConfig(
                group_col="suspect_key",
                event_col="is_anomaly",
                alpha=config.alpha,
            )
        ).rank(working)
        if not ranked:
            return []

        best = ranked[0]
        tool_id, chamber_id = split_suspect_key(best.group, config)
        metrics = {
            "p_value_combo": best.p_value,
            "p_value_combo_adjusted": best.p_value_adjusted,
            "num_hypotheses": best.metrics["num_hypotheses"],
            "multiple_testing_method": 1.0,
            "odds_ratio": best.odds_ratio,
            "sample_size": best.metrics["sample_size"],
            "fisher_fallback": best.metrics["fisher_fallback"],
        }
        if "p_value_global" in best.metrics:
            metrics["p_value_global"] = best.metrics["p_value_global"]
        return [
            RootCauseCandidate(
                suspect_tool_id=tool_id,
                suspect_chamber_id=chamber_id,
                confidence_score=(1 - best.p_value) * 100,
                metrics=metrics,
            )
        ]
