import pandas as pd
import pytest

from analysis_core.validation import require_columns


def test_passes_when_all_columns_present():
    require_columns(pd.DataFrame({"a": [1], "b": [2]}), ["a", "b"])


def test_raises_naming_missing_columns():
    with pytest.raises(ValueError, match=r"\['b', 'c'\]"):
        require_columns(pd.DataFrame({"a": [1]}), ["a", "b", "c"])
