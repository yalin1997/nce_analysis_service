import pandas as pd
from scipy import stats

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.base import DriftStrategy


class Correlation(DriftStrategy):
    def classify(
        self, series_df: pd.DataFrame, config: AnalysisConfig
    ) -> tuple[str, dict[str, float]]:
        ordered = series_df.sort_values("Pre_Execute_Time").reset_index(drop=True)
        timestamps = pd.to_datetime(ordered["Pre_Execute_Time"])
        elapsed_hours = (
            (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
        ).to_numpy()
        values = ordered["NCE_Value"].to_numpy()

        correlation, p_value = stats.pearsonr(elapsed_hours, values)
        metrics = {"correlation": float(correlation), "p_value": float(p_value)}

        if p_value < config.alpha and correlation > 0:
            return "CHAMBER_DRIFT", metrics
        return "SPECIFIC_CHAMBER_DEFECT", metrics
