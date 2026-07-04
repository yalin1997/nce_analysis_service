from abc import ABC, abstractmethod

import pandas as pd


class PreprocessingError(Exception):
    """Raised when input data does not match the expected wide-history schema."""


class PreprocessingStrategy(ABC):
    @abstractmethod
    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Reshape wide-format wafer records (with Measurement_Points and
        positionally-suffixed Pre_*_i history columns) into the standard long
        format with columns: PartID, WaferID, StageID, StepID, ToolID, ChuckID,
        Execute_Time, X_Posi, Y_Posi, NCE_Value, Pre_StageID, Pre_StepID,
        Pre_ToolID, Pre_ChamberID, Pre_Execute_Time."""
