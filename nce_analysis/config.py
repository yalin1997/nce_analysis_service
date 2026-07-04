from typing import Literal

from pydantic import BaseModel


class AnalysisConfig(BaseModel):
    spec_threshold: float = 15.0
    min_wafer_count: int = 5
    hotspot_ratio_threshold: float = 0.05
    noise_filter_majority_threshold: float = 0.5
    root_cause_strategy: Literal["statistical", "ml", "both"] = "both"
    drift_strategy: Literal["regression_cusum", "correlation"] = "regression_cusum"
    alpha: float = 0.05
    summary_top_n: int = 5
