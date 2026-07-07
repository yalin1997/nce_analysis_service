import pandas as pd
from pydantic import BaseModel, ConfigDict

from analysis_core.validation import require_columns


class RatioHotspotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_cols: list[str]
    event_col: str = "is_event"
    ratio_threshold: float = 0.05
    min_count: int = 5


class RatioHotspotDetector:
    """Group rows by group_cols and keep groups whose event ratio meets
    ratio_threshold. Groups below min_count rows are kept but flagged
    insufficient_sample=True so callers can report them without scoring."""

    def __init__(self, config: RatioHotspotConfig):
        self.config = config

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        config = self.config
        require_columns(df, [*config.group_cols, config.event_col])

        grouped = (
            df.groupby(config.group_cols)
            .agg(
                event_count=(config.event_col, "sum"),
                total_count=(config.event_col, "count"),
            )
            .reset_index()
        )
        grouped["event_ratio"] = grouped["event_count"] / grouped["total_count"]

        grouped = grouped[grouped["event_ratio"] >= config.ratio_threshold]
        grouped["insufficient_sample"] = grouped["total_count"] < config.min_count

        return grouped.sort_values(
            ["event_ratio", "event_count"], ascending=False
        ).reset_index(drop=True)
