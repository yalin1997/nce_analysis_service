import logging
import re
from collections import defaultdict

import pandas as pd

from nce_analysis.preprocessing.base import PreprocessingError, PreprocessingStrategy

logger = logging.getLogger(__name__)

HISTORY_FIELD_NAMES = ["StageID", "StepID", "ToolID", "ChamberID", "Execute_Time"]
_HISTORY_COLUMN_PATTERN = re.compile(
    r"^Pre_(StageID|StepID|ToolID|ChamberID|Execute_Time)_(\d+)$"
)
_REQUIRED_POINT_FIELDS = ("X_Posi", "Y_Posi", "NCE_Value")


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
    for wafer_id, points in zip(kept["WaferID"], kept["Measurement_Points"]):
        for point in points:
            if not isinstance(point, dict):
                raise PreprocessingError(
                    f"Wafer {wafer_id} has a malformed Measurement_Points entry "
                    f"(expected a dict, got {type(point).__name__}): {point!r}"
                )
            missing = [field for field in _REQUIRED_POINT_FIELDS if field not in point]
            if missing:
                raise PreprocessingError(
                    f"Wafer {wafer_id} has a Measurement_Points entry missing "
                    f"required field(s) {missing}: {point}"
                )
    exploded = kept.explode("Measurement_Points", ignore_index=True)
    points = pd.json_normalize(exploded["Measurement_Points"].tolist())
    return pd.concat(
        [exploded.drop(columns=["Measurement_Points"]).reset_index(drop=True), points],
        axis=1,
    )


class WideHistoryReshape(PreprocessingStrategy):
    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        levels = discover_history_levels(raw_df.columns)
        if not levels:
            raise PreprocessingError(
                "No Pre_<Field>_<N> history columns found in input data; "
                "expected at least one upstream history level."
            )

        exploded = explode_measurement_points(raw_df)
        base_columns = [
            c for c in exploded.columns if not _HISTORY_COLUMN_PATTERN.match(c)
        ]

        level_frames = []
        for field_map in levels.values():
            level_df = exploded[base_columns].copy()
            for field_name in HISTORY_FIELD_NAMES:
                source_col = field_map.get(field_name)
                canonical_col = f"Pre_{field_name}"
                level_df[canonical_col] = exploded[source_col] if source_col else pd.NA
            level_df = level_df[level_df["Pre_StageID"].notna()]
            level_frames.append(level_df)

        return pd.concat(level_frames, ignore_index=True)
