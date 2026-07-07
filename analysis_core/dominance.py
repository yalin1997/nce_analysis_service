from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ConfigDict


class DominanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: float = 0.5


@dataclass
class Dominance:
    category: tuple
    share: float
    count: int


def find_dominant(keys: pd.Series, config: DominanceConfig) -> Dominance | None:
    """Return the most frequent category if its share of `keys` is strictly
    greater than config.threshold, else None. Scalar categories are wrapped
    as 1-tuples so composite keys (already tuples) and scalars have a
    uniform return shape."""
    if keys.empty:
        return None
    counts = keys.value_counts()
    top_category, top_count = counts.index[0], int(counts.iloc[0])
    share = top_count / len(keys)
    if share <= config.threshold:
        return None
    category = top_category if isinstance(top_category, tuple) else (top_category,)
    return Dominance(category=category, share=share, count=top_count)
