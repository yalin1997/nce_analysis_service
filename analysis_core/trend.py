from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import stats

from analysis_core.validation import require_columns


class TrendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_col: str = "timestamp"
    value_col: str = "value"
    alpha: float = 0.05
    cusum_sigma: float = 3.0


@dataclass
class TrendResult:
    kind: Literal["change_point", "increasing_trend", "none"]
    metrics: dict[str, float] = field(default_factory=dict)


def _elapsed_hours_and_values(
    df: pd.DataFrame, config: TrendConfig
) -> tuple[np.ndarray, np.ndarray]:
    require_columns(df, [config.time_col, config.value_col])
    ordered = df.sort_values(config.time_col).reset_index(drop=True)
    timestamps = pd.to_datetime(ordered[config.time_col])
    elapsed_hours = (
        (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
    ).to_numpy()
    values = ordered[config.value_col].to_numpy()
    return elapsed_hours, values


class RegressionCusumDetector:
    """Linear regression for a monotone trend plus a CUSUM control-chart pass
    for abrupt change points. change_point outranks increasing_trend."""

    def __init__(self, config: TrendConfig):
        self.config = config

    def detect(self, df: pd.DataFrame) -> TrendResult:
        elapsed_hours, values = _elapsed_hours_and_values(df, self.config)

        if np.unique(elapsed_hours).size < 2:
            return TrendResult("none", {"insufficient_time_variation": 1.0})

        slope, intercept, _, p_value, _ = stats.linregress(elapsed_hours, values)
        fitted = intercept + slope * elapsed_hours
        residuals = values - fitted

        cusum = np.cumsum(residuals - residuals.mean())
        cusum_range = float(cusum.max() - cusum.min())
        residual_std = float(residuals.std())
        n = len(residuals)
        # Threshold scales with sample size so short, noisy series aren't
        # flagged as sudden shifts when they're really just noise.
        cusum_threshold = (
            self.config.cusum_sigma * residual_std * np.sqrt(n)
            if residual_std > 0 and n > 0
            else 0.0
        )
        change_point_detected = cusum_threshold > 0 and cusum_range > cusum_threshold

        metrics = {
            "slope": float(slope),
            "slope_p_value": float(p_value),
            "cusum_range": cusum_range,
            "cusum_threshold": cusum_threshold,
            "cusum_threshold_method": 1.0,
        }

        if change_point_detected:
            return TrendResult("change_point", metrics)
        if p_value < self.config.alpha and slope > 0:
            return TrendResult("increasing_trend", metrics)
        return TrendResult("none", metrics)


class CorrelationTrendDetector:
    """Pearson correlation of value against elapsed time."""

    def __init__(self, config: TrendConfig):
        self.config = config

    def detect(self, df: pd.DataFrame) -> TrendResult:
        elapsed_hours, values = _elapsed_hours_and_values(df, self.config)

        if np.unique(elapsed_hours).size < 2:
            return TrendResult("none", {"insufficient_time_variation": 1.0})
        if np.unique(values).size < 2:
            return TrendResult("none", {"insufficient_value_variation": 1.0})

        correlation, p_value = stats.pearsonr(elapsed_hours, values)
        metrics = {"correlation": float(correlation), "p_value": float(p_value)}

        if p_value < self.config.alpha and correlation > 0:
            return TrendResult("increasing_trend", metrics)
        return TrendResult("none", metrics)
