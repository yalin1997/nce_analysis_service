import json
from pathlib import Path

import pandas as pd


class InputFormatError(Exception):
    """Raised when an input file's on-disk shape can't be mapped to the
    Measurement_Points-per-wafer contract pipeline.run expects."""


_LONG_POINT_COLUMNS = {"X_Posi", "Y_Posi", "NCE_Value"}


def load_input(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _normalize_csv(pd.read_csv(path))
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        df["Measurement_Points"] = df["Measurement_Points"].map(list)
        return df

    raise InputFormatError(
        f"Unrecognized input file extension {suffix!r} (expected .csv, .parquet, or .pq)"
    )


def _normalize_csv(df: pd.DataFrame) -> pd.DataFrame:
    if "Measurement_Points" in df.columns:
        df = df.copy()
        df["Measurement_Points"] = df["Measurement_Points"].map(json.loads)
        return df

    if _LONG_POINT_COLUMNS.issubset(df.columns):
        return _reconstruct_from_long(df)

    raise InputFormatError(
        "CSV must have either a 'Measurement_Points' column (JSON-encoded "
        "list per row) or 'X_Posi'/'Y_Posi'/'NCE_Value' columns (one row "
        "per measurement point)."
    )


def _reconstruct_from_long(df: pd.DataFrame) -> pd.DataFrame:
    if "WaferID" not in df.columns:
        raise InputFormatError(
            "Long-format CSV requires a 'WaferID' column to group "
            "measurement points back into per-wafer rows."
        )
    wafer_cols = [c for c in df.columns if c not in _LONG_POINT_COLUMNS]
    point_cols = sorted(_LONG_POINT_COLUMNS)

    rows = []
    for _, group in df.groupby("WaferID", sort=False):
        row = group.iloc[0][wafer_cols].to_dict()
        row["Measurement_Points"] = group[point_cols].to_dict("records")
        rows.append(row)
    return pd.DataFrame(rows)
