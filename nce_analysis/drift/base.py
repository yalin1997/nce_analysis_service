from abc import ABC, abstractmethod

import pandas as pd

from nce_analysis.config import AnalysisConfig


class DriftStrategy(ABC):
    @abstractmethod
    def classify(
        self, series_df: pd.DataFrame, config: AnalysisConfig
    ) -> tuple[str, dict[str, float]]:
        """series_df: columns Pre_Execute_Time (datetime-parseable), NCE_Value
        for all wafers processed by one suspect Pre_ToolID+Pre_ChamberID
        combination within a single (Pre_StageID, Pre_StepID) group at the
        affected coordinate. Returns
        (root_cause_type, metrics) where root_cause_type is one of
        'SPECIFIC_CHAMBER_DEFECT', 'CHAMBER_DRIFT', 'CHAMBER_SUDDEN_SHIFT'."""


KIND_TO_ROOT_CAUSE = {
    "change_point": "CHAMBER_SUDDEN_SHIFT",
    "increasing_trend": "CHAMBER_DRIFT",
    "none": "SPECIFIC_CHAMBER_DEFECT",
}
