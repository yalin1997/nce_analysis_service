import logging

import pandas as pd

from nce_analysis.preprocessing.base import PreprocessingError

logger = logging.getLogger(__name__)


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
