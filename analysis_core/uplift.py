from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, ConfigDict

from analysis_core.validation import require_columns


class UpliftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_col: str = "group"
    event_col: str = "is_event"


@dataclass
class GroupUplift:
    group: str
    uplift: float
    normalized_share: float
    metrics: dict[str, float] = field(default_factory=dict)


class RiskUpliftRanker:
    """Score each group by its event-rate uplift over the overall event rate.
    Returns only positive-uplift groups, highest uplift first; empty list
    means no group is elevated (or the labels are degenerate)."""

    def __init__(self, config: UpliftConfig):
        self.config = config

    def rank(self, df: pd.DataFrame) -> list[GroupUplift]:
        config = self.config
        require_columns(df, [config.group_col, config.event_col])

        if df[config.group_col].nunique() < 2:
            return []

        labels = df[config.event_col].astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            return []

        overall_rate = float(labels.mean())
        grouped = (
            df.assign(_label=labels)
            .groupby(config.group_col)["_label"]
            .agg(["mean", "sum", "count"])
        )
        grouped["uplift"] = grouped["mean"] - overall_rate
        positive = grouped[grouped["uplift"] > 0]
        if positive.empty:
            return []

        positive_total = float(positive["uplift"].sum())
        results = [
            GroupUplift(
                group=group,
                uplift=float(row["uplift"]),
                normalized_share=(
                    float(row["uplift"]) / positive_total if positive_total > 0 else 0.0
                ),
                metrics={
                    "group_event_rate": float(row["mean"]),
                    "overall_event_rate": overall_rate,
                    "group_event_count": float(row["sum"]),
                    "group_size": float(row["count"]),
                    "sample_size": float(len(df)),
                },
            )
            for group, row in positive.iterrows()
        ]
        # Stable sort keeps groupby-index order on ties, matching the old
        # idxmax()-based single-winner selection exactly.
        results.sort(key=lambda item: item.uplift, reverse=True)
        return results
