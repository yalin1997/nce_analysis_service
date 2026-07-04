import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.hotspot.base import HotspotStrategy

LITHO_POINT_KEY_COLUMNS = ["WaferID", "X_Posi", "Y_Posi"]


class RatioThreshold(HotspotStrategy):
    def detect(self, long_df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
        litho_points = long_df.drop_duplicates(subset=LITHO_POINT_KEY_COLUMNS).copy()
        litho_points["is_anomaly"] = litho_points["NCE_Value"] > config.spec_threshold

        grouped = (
            litho_points.groupby(["X_Posi", "Y_Posi"])
            .agg(
                anomalous_wafer_count=("is_anomaly", "sum"),
                total_wafer_count=("is_anomaly", "count"),
            )
            .reset_index()
        )
        grouped["anomaly_ratio"] = (
            grouped["anomalous_wafer_count"] / grouped["total_wafer_count"]
        )

        grouped = grouped[grouped["anomaly_ratio"] >= config.hotspot_ratio_threshold]
        grouped["insufficient_sample"] = (
            grouped["total_wafer_count"] < config.min_wafer_count
        )

        return grouped.sort_values(
            ["anomaly_ratio", "anomalous_wafer_count"], ascending=False
        ).reset_index(drop=True)
