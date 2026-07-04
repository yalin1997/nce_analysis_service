from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.schema import RootCauseDetail


@dataclass
class NoiseFilterResult:
    surviving_hotspots: pd.DataFrame
    litho_self_issues: list[RootCauseDetail]


class NoiseFilterStrategy(ABC):
    @abstractmethod
    def filter(
        self, long_df: pd.DataFrame, hotspots: pd.DataFrame, config: AnalysisConfig
    ) -> NoiseFilterResult:
        """hotspots: output of a HotspotStrategy.detect(). Returns a
        NoiseFilterResult splitting hotspots into ones attributable to LITHO's
        own Tool/Chuck (litho_self_issues, already-classified RootCauseDetail
        objects) and ones that survive to be analyzed for upstream root cause
        (surviving_hotspots, same shape as the `hotspots` input)."""
