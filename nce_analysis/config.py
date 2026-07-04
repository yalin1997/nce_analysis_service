from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_threshold: float = 15.0
    min_wafer_count: int = Field(default=5, ge=1)
    hotspot_ratio_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    noise_filter_majority_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    root_cause_strategy: Literal["statistical", "ml", "both"] = "both"
    root_cause_granularity: Literal["tool", "chamber"] = "chamber"
    drift_strategy: Literal["regression_cusum", "correlation"] = "regression_cusum"
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    summary_top_n: int = Field(default=5, ge=1)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalysisConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)
