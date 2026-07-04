import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import RootCauseCandidate, RootCauseStrategy
from nce_analysis.root_cause.ml import MLStrategy
from nce_analysis.root_cause.statistical import StatisticalStrategy


class BothStrategy(RootCauseStrategy):
    def __init__(self):
        self._statistical = StatisticalStrategy()
        self._ml = MLStrategy()

    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        statistical_results = self._statistical.analyze(group_df, config)
        ml_results = self._ml.analyze(group_df, config)

        if not statistical_results and not ml_results:
            return []
        if not statistical_results:
            for candidate in ml_results:
                candidate.requires_manual_review = True
            return ml_results
        if not ml_results:
            for candidate in statistical_results:
                candidate.requires_manual_review = True
            return statistical_results

        stat_candidate = statistical_results[0]
        ml_candidate = ml_results[0]
        agree = (
            stat_candidate.suspect_tool_id == ml_candidate.suspect_tool_id
            and stat_candidate.suspect_chamber_id == ml_candidate.suspect_chamber_id
        )
        merged_metrics = {
            **{f"statistical_{k}": v for k, v in stat_candidate.metrics.items()},
            **{f"ml_{k}": v for k, v in ml_candidate.metrics.items()},
        }

        if agree:
            return [
                RootCauseCandidate(
                    suspect_tool_id=stat_candidate.suspect_tool_id,
                    suspect_chamber_id=stat_candidate.suspect_chamber_id,
                    confidence_score=max(
                        stat_candidate.confidence_score, ml_candidate.confidence_score
                    ),
                    metrics=merged_metrics,
                    requires_manual_review=False,
                )
            ]

        stat_candidate.requires_manual_review = True
        ml_candidate.requires_manual_review = True
        return [stat_candidate, ml_candidate]
