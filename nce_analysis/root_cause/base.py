from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from nce_analysis.config import AnalysisConfig


@dataclass
class RootCauseCandidate:
    suspect_tool_id: str
    suspect_chamber_id: str
    confidence_score: float
    metrics: dict[str, float] = field(default_factory=dict)
    requires_manual_review: bool = False


class RootCauseStrategy(ABC):
    @abstractmethod
    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        """group_df: rows for wafers at one hotspot coordinate and a single
        (Pre_StageID, Pre_StepID) group, with columns Pre_ToolID,
        Pre_ChamberID, is_anomaly.
        Returns a list of suspect Pre_ToolID+Pre_ChamberID candidates (usually
        0 or 1 entries; the 'both' cross-validation strategy may return 2 when
        its sub-strategies disagree, each marked requires_manual_review=True)."""
