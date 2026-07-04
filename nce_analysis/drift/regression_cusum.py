import numpy as np
import pandas as pd
from scipy import stats

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.base import DriftStrategy


class RegressionCusum(DriftStrategy):
    def classify(
        self, series_df: pd.DataFrame, config: AnalysisConfig
    ) -> tuple[str, dict[str, float]]:
        ordered = series_df.sort_values("Pre_Execute_Time").reset_index(drop=True)
        timestamps = pd.to_datetime(ordered["Pre_Execute_Time"])
        elapsed_hours = (
            (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
        ).to_numpy()
        values = ordered["NCE_Value"].to_numpy()

        slope, intercept, _, p_value, _ = stats.linregress(elapsed_hours, values)
        fitted = intercept + slope * elapsed_hours
        residuals = values - fitted

        cusum = np.cumsum(residuals - residuals.mean())
        cusum_range = float(cusum.max() - cusum.min())
        residual_std = float(residuals.std())
        cusum_threshold = 5 * residual_std if residual_std > 0 else 0.0
        change_point_detected = cusum_threshold > 0 and cusum_range > cusum_threshold

        metrics = {
            "slope": float(slope),
            "slope_p_value": float(p_value),
            "cusum_range": cusum_range,
            "cusum_threshold": cusum_threshold,
        }

        if change_point_detected:
            return "CHAMBER_SUDDEN_SHIFT", metrics
        if p_value < config.alpha and slope > 0:
            return "CHAMBER_DRIFT", metrics
        return "SPECIFIC_CHAMBER_DEFECT", metrics
