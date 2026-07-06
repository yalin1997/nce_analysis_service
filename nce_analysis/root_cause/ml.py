import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import (
    RootCauseCandidate,
    RootCauseStrategy,
    build_suspect_key,
    split_suspect_key,
)


class MLStrategy(RootCauseStrategy):
    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        working = group_df.copy()
        working["suspect_key"] = build_suspect_key(working, config)

        if working["suspect_key"].nunique() < 2:
            return []

        labels = working["is_anomaly"].astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            return []

        overall_rate = float(labels.mean())
        grouped = (
            working.assign(_label=labels)
            .groupby("suspect_key")["_label"]
            .agg(["mean", "sum", "count"])
        )
        grouped["risk_uplift"] = grouped["mean"] - overall_rate
        positive = grouped[grouped["risk_uplift"] > 0].copy()
        if positive.empty:
            return []

        best_suspect_key = positive["risk_uplift"].idxmax()
        best = positive.loc[best_suspect_key]
        positive_total = float(positive["risk_uplift"].sum())
        confidence_score = (
            float(best["risk_uplift"]) / positive_total * 100.0
            if positive_total > 0
            else 0.0
        )

        tool_id, chamber_id = split_suspect_key(best_suspect_key, config)
        return [
            RootCauseCandidate(
                suspect_tool_id=tool_id,
                suspect_chamber_id=chamber_id,
                confidence_score=float(confidence_score),
                metrics={
                    "risk_uplift": float(best["risk_uplift"]),
                    "suspect_anomaly_rate": float(best["mean"]),
                    "overall_anomaly_rate": overall_rate,
                    "suspect_anomaly_count": float(best["sum"]),
                    "suspect_sample_size": float(best["count"]),
                    "sample_size": float(len(working)),
                    "ml_scoring_method": 1.0,
                },
            )
        ]
