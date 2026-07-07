import pandas as pd

from analysis_core import RegressionCusumDetector, TrendConfig

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.base import KIND_TO_ROOT_CAUSE, DriftStrategy


class RegressionCusum(DriftStrategy):
    def classify(
        self, series_df: pd.DataFrame, config: AnalysisConfig
    ) -> tuple[str, dict[str, float]]:
        result = RegressionCusumDetector(
            TrendConfig(
                time_col="Pre_Execute_Time",
                value_col="NCE_Value",
                alpha=config.alpha,
            )
        ).detect(series_df)
        return KIND_TO_ROOT_CAUSE[result.kind], result.metrics
