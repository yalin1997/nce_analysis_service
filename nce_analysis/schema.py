from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from nce_analysis.config import AnalysisConfig

RootCauseType = Literal[
    "LITHO_TOOL_ISSUE",
    "LITHO_CHUCK_ISSUE",
    "LITHO_CHUCK_CONTAMINATION",
    "SPECIFIC_CHAMBER_DEFECT",
    "CHAMBER_DRIFT",
    "CHAMBER_SUDDEN_SHIFT",
]


class InsufficientSampleHotspot(BaseModel):
    X_Posi: float
    Y_Posi: float
    anomaly_ratio: float
    anomalous_wafer_count: int
    total_wafer_count: int


class RootCauseDetail(BaseModel):
    Suspect_Pre_ToolID: str
    Suspect_Pre_ChamberID: str
    Suspect_Pre_StepID: str
    Confidence_Score: float
    Root_Cause_Type: RootCauseType
    Affected_Coordinates: list[tuple[float, float]]
    Metrics: dict[str, float]
    Requires_Manual_Review: bool = False


class AnalysisResult(BaseModel):
    Summary: list[RootCauseDetail]
    Details: list[RootCauseDetail]
    Insufficient_Sample_Hotspots: list[InsufficientSampleHotspot] = []
    Generated_At: datetime
    Config_Used: AnalysisConfig
