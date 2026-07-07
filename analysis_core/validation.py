from collections.abc import Sequence

import pandas as pd


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise ValueError naming every configured column missing from df."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required column(s): {missing}")
