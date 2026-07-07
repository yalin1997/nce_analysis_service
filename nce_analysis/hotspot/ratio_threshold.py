import pandas as pd

from analysis_core import RatioHotspotConfig, RatioHotspotDetector

from nce_analysis.config import AnalysisConfig
from nce_analysis.hotspot.base import HotspotStrategy

LITHO_POINT_KEY_COLUMNS = ["WaferID", "X_Posi", "Y_Posi"]


class RatioThreshold(HotspotStrategy):
    def detect(self, long_df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
        litho_points = long_df.drop_duplicates(subset=LITHO_POINT_KEY_COLUMNS).copy()
        litho_points["is_anomaly"] = litho_points["NCE_Value"] > config.spec_threshold

        detected = RatioHotspotDetector(
            RatioHotspotConfig(
                group_cols=["X_Posi", "Y_Posi"],
                event_col="is_anomaly",
                ratio_threshold=config.hotspot_ratio_threshold,
                min_count=config.min_wafer_count,
            )
        ).detect(litho_points)

        return detected.rename(
            columns={
                "event_count": "anomalous_wafer_count",
                "total_count": "total_wafer_count",
                "event_ratio": "anomaly_ratio",
            }
        )
