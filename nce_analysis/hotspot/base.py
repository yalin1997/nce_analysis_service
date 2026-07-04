from abc import ABC, abstractmethod

import pandas as pd

from nce_analysis.config import AnalysisConfig


class HotspotStrategy(ABC):
    @abstractmethod
    def detect(self, long_df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
        """long_df: standard long-format table (see preprocessing). Returns a
        DataFrame with columns X_Posi, Y_Posi, anomalous_wafer_count,
        total_wafer_count, anomaly_ratio, insufficient_sample (bool), sorted
        by anomaly_ratio desc, restricted to coordinates meeting
        config.hotspot_ratio_threshold. Coordinates below
        config.min_wafer_count are kept but flagged insufficient_sample=True:
        reported raw, excluded from root-cause analysis."""
