import logging
import re
from collections import defaultdict

import pandas as pd

from nce_analysis.preprocessing.base import PreprocessingError

logger = logging.getLogger(__name__)

HISTORY_FIELD_NAMES = ["StageID", "StepID", "ToolID", "ChamberID", "Execute_Time"]
_HISTORY_COLUMN_PATTERN = re.compile(
    r"^Pre_(StageID|StepID|ToolID|ChamberID|Execute_Time)_(\d+)$"
)


def discover_history_levels(columns) -> dict[int, dict[str, str]]:
    """Scan column names for Pre_<Field>_<N> groups and return, per level N,
    a mapping of canonical field name -> actual column name."""
    levels: dict[int, dict[str, str]] = defaultdict(dict)
    for col in columns:
        match = _HISTORY_COLUMN_PATTERN.match(col)
        if match:
            field_name, level_str = match.groups()
            levels[int(level_str)][field_name] = col
    return dict(levels)


def explode_measurement_points(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Explode the Measurement_Points column (list[dict] per row) into one row
    per (X_Posi, Y_Posi, NCE_Value), carrying every other column along.
    Wafers with an empty/missing Measurement_Points list are skipped with a
    logged warning; if every wafer is empty, raise PreprocessingError."""
    has_points = raw_df["Measurement_Points"].map(
        lambda points: isinstance(points, list) and len(points) > 0
    )
    if not has_points.all():
        skipped = raw_df.loc[~has_points, "WaferID"].tolist()
        logger.warning(
            "Skipping %d wafer(s) with empty Measurement_Points: %s",
            len(skipped),
            skipped,
        )
    if not has_points.any():
        raise PreprocessingError(
            "Every wafer in the batch has an empty Measurement_Points list; "
            "nothing to analyze."
        )
    kept = raw_df[has_points]
    exploded = kept.explode("Measurement_Points", ignore_index=True)
    points = pd.json_normalize(exploded["Measurement_Points"].tolist())
    return pd.concat(
        [exploded.drop(columns=["Measurement_Points"]).reset_index(drop=True), points],
        axis=1,
    )
