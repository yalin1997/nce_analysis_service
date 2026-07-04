# Cross-Stage Impact Analysis Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-Python library that analyzes LITHO NCE measurement hotspots and traces them back to the upstream (CMP/CVD/PVD) tool+chamber most likely responsible, filtering out anomalies actually caused by the LITHO tool/chuck itself.

**Architecture:** A pipeline of swappable strategy components (Preprocessing → Hotspot Detection → Noise Filter → Root Cause Analysis → Drift Analysis → Result Aggregation), each stage implementing a common ABC interface and selected at runtime via a single `AnalysisConfig`. Per-upstream-step grouping is done by grouping on the actual `(Pre_StageID, Pre_StepID)` combination discovered during preprocessing, not by positional column suffix — repeated passes through the same stage type (StepIDs like `3580.01` vs `3580.02`) are analyzed as separate groups, so each wafer appears exactly once per group.

**Tech Stack:** Python 3.11+, pandas, pydantic v2, scipy (chi-square/Fisher/linregress/pearsonr), scikit-learn (DecisionTreeClassifier), shap, pytest. (XGBoost is a future strategy variant — deliberately not a v1 dependency.)

## Global Constraints

- Pure Python library — no REST/API layer (per spec §1).
- Input scale: hundreds to low-thousands of wafers per call (per spec §1) — no need to optimize for millions of rows.
- `X_Posi`/`Y_Posi` are already on a fixed 6mm grid, directly comparable across wafers — no spatial tolerance/binning logic (per spec §2).
- History levels `Pre_*_1..N` have variable, per-dataset `N`; the positional suffix does **not** determine stage type — root-cause grouping must use the actual `(Pre_StageID, Pre_StepID)` combination (per spec §2, §3, §4). A wafer may pass the same stage type more than once; `StepID` distinguishes the passes via its decimal suffix (e.g. `3580.01`, `3580.02`), so neither pass's tool/chamber record may be dropped.
- Residual duplicates within one `(WaferID, Pre_StageID, Pre_StepID)` group (rework/data duplication) keep the most recent `Pre_Execute_Time` record and log a warning (per spec §11).
- Default config values (per spec §10): `spec_threshold=15.0`, `min_wafer_count=5`, `hotspot_ratio_threshold=0.05`, `noise_filter_majority_threshold=0.5`, `root_cause_strategy="both"`, `drift_strategy="regression_cusum"`, `alpha=0.05`, `summary_top_n=5`.
- `PreprocessingError` must be raised (not silently swallowed) when no `Pre_*_i` columns are found (per spec §11).
- Wafers with an empty `Measurement_Points` list are skipped with a logged warning; an all-empty batch raises `PreprocessingError` (per spec §11).
- Hotspots below `min_wafer_count` are never given a confidence score — they are reported raw in `AnalysisResult.Insufficient_Sample_Hotspots` (per spec §5, §11).
- `PRODUCT_TOOL_MISMATCH` / `PartID` dimension is explicitly out of scope for v1 (per spec §9) — do not add it.
- Validate only at system boundaries; trust internal data once past them (per spec §11).

---

## Task 1: Project Scaffolding + AnalysisConfig

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `nce_analysis/__init__.py`
- Create: `nce_analysis/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `AnalysisConfig` (pydantic `BaseModel`) with fields `spec_threshold: float`, `min_wafer_count: int`, `hotspot_ratio_threshold: float`, `noise_filter_majority_threshold: float`, `root_cause_strategy: Literal["statistical","ml","both"]`, `drift_strategy: Literal["regression_cusum","correlation"]`, `alpha: float`, `summary_top_n: int`. Every later task imports this from `nce_analysis.config`.

- [ ] **Step 1: Create the project files**

`pyproject.toml`:
```toml
[project]
name = "nce-analysis"
version = "0.1.0"
description = "Cross-stage impact analysis service for LITHO NCE root-cause analysis"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "pydantic>=2.0",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "shap>=0.44",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["nce_analysis"]
```

`.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
```

`nce_analysis/__init__.py`:
```python
```
(empty file — package marker)

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:
```python
import pytest
from pydantic import ValidationError

from nce_analysis.config import AnalysisConfig


def test_default_config_values():
    config = AnalysisConfig()
    assert config.spec_threshold == 15.0
    assert config.min_wafer_count == 5
    assert config.hotspot_ratio_threshold == 0.05
    assert config.noise_filter_majority_threshold == 0.5
    assert config.root_cause_strategy == "both"
    assert config.drift_strategy == "regression_cusum"
    assert config.alpha == 0.05
    assert config.summary_top_n == 5


def test_config_rejects_invalid_root_cause_strategy():
    with pytest.raises(ValidationError):
        AnalysisConfig(root_cause_strategy="invalid_choice")


def test_config_rejects_invalid_drift_strategy():
    with pytest.raises(ValidationError):
        AnalysisConfig(drift_strategy="invalid_choice")
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
cd /mnt/c/Users/y4lin/gemini_repo/nce_analysis_service
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_config.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.config'`

- [ ] **Step 4: Write minimal implementation**

`nce_analysis/config.py`:
```python
from typing import Literal

from pydantic import BaseModel


class AnalysisConfig(BaseModel):
    spec_threshold: float = 15.0
    min_wafer_count: int = 5
    hotspot_ratio_threshold: float = 0.05
    noise_filter_majority_threshold: float = 0.5
    root_cause_strategy: Literal["statistical", "ml", "both"] = "both"
    drift_strategy: Literal["regression_cusum", "correlation"] = "regression_cusum"
    alpha: float = 0.05
    summary_top_n: int = 5
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore nce_analysis/__init__.py nce_analysis/config.py tests/test_config.py
git commit -m "feat: add project scaffolding and AnalysisConfig"
```

---

## Task 2: Output Schema

**Files:**
- Create: `nce_analysis/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: `AnalysisConfig` from `nce_analysis.config`
- Produces: `RootCauseDetail` (pydantic model), `InsufficientSampleHotspot` (pydantic model), `AnalysisResult` (pydantic model), `RootCauseType` (Literal alias). Used by `result.py` and `pipeline.py` in later tasks.

- [ ] **Step 1: Write the failing test**

`tests/test_schema.py`:
```python
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from nce_analysis.config import AnalysisConfig
from nce_analysis.schema import AnalysisResult, RootCauseDetail


def _detail(**overrides):
    defaults = dict(
        Suspect_Pre_ToolID="CMP_01",
        Suspect_Pre_ChamberID="ChamberA",
        Suspect_Pre_StepID="3580.01",
        Confidence_Score=92.5,
        Root_Cause_Type="SPECIFIC_CHAMBER_DEFECT",
        Affected_Coordinates=[(0.0, 0.0)],
        Metrics={"p_value": 0.001, "odds_ratio": 8.2},
    )
    defaults.update(overrides)
    return RootCauseDetail(**defaults)


def test_root_cause_detail_defaults_requires_manual_review_false():
    detail = _detail()
    assert detail.Requires_Manual_Review is False


def test_root_cause_detail_rejects_invalid_root_cause_type():
    with pytest.raises(ValidationError):
        _detail(Root_Cause_Type="NOT_A_REAL_TYPE")


def test_analysis_result_construction():
    detail = _detail()
    result = AnalysisResult(
        Summary=[detail],
        Details=[detail],
        Generated_At=datetime.now(timezone.utc),
        Config_Used=AnalysisConfig(),
    )
    assert result.Summary[0].Suspect_Pre_ToolID == "CMP_01"
    assert result.Config_Used.spec_threshold == 15.0
    assert result.Insufficient_Sample_Hotspots == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.schema'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/schema.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/schema.py tests/test_schema.py
git commit -m "feat: add input/output pydantic schema"
```

---

## Task 3: Preprocessing Base Interface

**Files:**
- Create: `nce_analysis/preprocessing/__init__.py`
- Create: `nce_analysis/preprocessing/base.py`
- Test: `tests/test_preprocessing_base.py`

**Interfaces:**
- Produces: `PreprocessingStrategy` (ABC with abstract `transform(raw_df: pd.DataFrame) -> pd.DataFrame`), `PreprocessingError` (Exception subclass). Used by `wide_history_reshape.py` (Task 6) and `pipeline.py` (Task 19).

- [ ] **Step 1: Write the failing test**

`tests/test_preprocessing_base.py`:
```python
import pandas as pd
import pytest

from nce_analysis.preprocessing.base import PreprocessingStrategy


def test_preprocessing_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        PreprocessingStrategy()


def test_concrete_subclass_can_implement_transform():
    class DummyStrategy(PreprocessingStrategy):
        def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
            return raw_df

    strategy = DummyStrategy()
    df = pd.DataFrame({"a": [1, 2]})
    result = strategy.transform(df)
    pd.testing.assert_frame_equal(result, df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preprocessing_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.preprocessing'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/preprocessing/__init__.py`:
```python
```
(empty file — package marker)

`nce_analysis/preprocessing/base.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_preprocessing_base.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/preprocessing/__init__.py nce_analysis/preprocessing/base.py tests/test_preprocessing_base.py
git commit -m "feat: add PreprocessingStrategy interface and PreprocessingError"
```

---

## Task 4: Measurement Point Explosion Helper

**Files:**
- Create: `nce_analysis/preprocessing/wide_history_reshape.py`
- Test: `tests/test_wide_history_reshape.py`

**Interfaces:**
- Consumes: `PreprocessingError` from `nce_analysis.preprocessing.base` (Task 3)
- Produces: `explode_measurement_points(raw_df: pd.DataFrame) -> pd.DataFrame` — module-level function. Skips wafers with empty `Measurement_Points` (logged warning); raises `PreprocessingError` if every wafer is empty. Used internally by `WideHistoryReshape.transform` (Task 6).

- [ ] **Step 1: Write the failing test**

`tests/test_wide_history_reshape.py`:
```python
import pandas as pd
import pytest

from nce_analysis.preprocessing.base import PreprocessingError
from nce_analysis.preprocessing.wide_history_reshape import explode_measurement_points


def test_explode_measurement_points_expands_rows():
    raw_df = pd.DataFrame(
        [
            {
                "WaferID": "W1",
                "ToolID": "LITHO_01",
                "Measurement_Points": [
                    {"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 10.0},
                    {"X_Posi": 6.0, "Y_Posi": 0.0, "NCE_Value": 20.0},
                ],
            },
            {
                "WaferID": "W2",
                "ToolID": "LITHO_02",
                "Measurement_Points": [
                    {"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 5.0},
                ],
            },
        ]
    )

    result = explode_measurement_points(raw_df)

    assert len(result) == 3
    assert set(result.columns) == {"WaferID", "ToolID", "X_Posi", "Y_Posi", "NCE_Value"}
    assert result.loc[result["WaferID"] == "W2", "NCE_Value"].iloc[0] == 5.0
    assert (result["WaferID"] == "W1").sum() == 2


def test_explode_skips_wafers_with_empty_measurement_points(caplog):
    raw_df = pd.DataFrame(
        [
            {
                "WaferID": "W1",
                "ToolID": "LITHO_01",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 10.0}],
            },
            {"WaferID": "W2", "ToolID": "LITHO_02", "Measurement_Points": []},
        ]
    )

    with caplog.at_level("WARNING"):
        result = explode_measurement_points(raw_df)

    assert result["WaferID"].tolist() == ["W1"]
    assert "W2" in caplog.text


def test_explode_raises_when_all_wafers_have_empty_measurement_points():
    raw_df = pd.DataFrame([{"WaferID": "W1", "Measurement_Points": []}])

    with pytest.raises(PreprocessingError):
        explode_measurement_points(raw_df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wide_history_reshape.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.preprocessing.wide_history_reshape'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/preprocessing/wide_history_reshape.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wide_history_reshape.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/preprocessing/wide_history_reshape.py tests/test_wide_history_reshape.py
git commit -m "feat: add explode_measurement_points helper"
```

---

## Task 5: History Level Discovery Helper

**Files:**
- Modify: `nce_analysis/preprocessing/wide_history_reshape.py`
- Modify: `tests/test_wide_history_reshape.py`

**Interfaces:**
- Produces: `discover_history_levels(columns) -> dict[int, dict[str, str]]` — module-level function. Used internally by `WideHistoryReshape.transform` (Task 6). Return shape: `{level_int: {field_name: actual_column_name}}`, e.g. `{1: {"StageID": "Pre_StageID_1", ...}, 2: {...}}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wide_history_reshape.py`:
```python
from nce_analysis.preprocessing.wide_history_reshape import discover_history_levels


def test_discover_history_levels_finds_all_levels():
    columns = [
        "WaferID",
        "X_Posi",
        "Pre_StageID_1",
        "Pre_StepID_1",
        "Pre_ToolID_1",
        "Pre_ChamberID_1",
        "Pre_Execute_Time_1",
        "Pre_StageID_2",
        "Pre_StepID_2",
        "Pre_ToolID_2",
        "Pre_ChamberID_2",
        "Pre_Execute_Time_2",
    ]

    levels = discover_history_levels(columns)

    assert set(levels.keys()) == {1, 2}
    assert levels[1]["StageID"] == "Pre_StageID_1"
    assert levels[1]["ToolID"] == "Pre_ToolID_1"
    assert levels[2]["ChamberID"] == "Pre_ChamberID_2"


def test_discover_history_levels_returns_empty_when_no_match():
    columns = ["WaferID", "X_Posi", "NCE_Value"]

    levels = discover_history_levels(columns)

    assert levels == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wide_history_reshape.py -v`
Expected: FAIL with `ImportError: cannot import name 'discover_history_levels'`

- [ ] **Step 3: Write minimal implementation**

Add to `nce_analysis/preprocessing/wide_history_reshape.py` (above the existing `explode_measurement_points` function):
```python
import re
from collections import defaultdict

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wide_history_reshape.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/preprocessing/wide_history_reshape.py tests/test_wide_history_reshape.py
git commit -m "feat: add discover_history_levels helper"
```

---

## Task 6: WideHistoryReshape Full Transform

**Files:**
- Modify: `nce_analysis/preprocessing/wide_history_reshape.py`
- Modify: `tests/test_wide_history_reshape.py`

**Interfaces:**
- Consumes: `PreprocessingStrategy`, `PreprocessingError` from `nce_analysis.preprocessing.base`; `explode_measurement_points`, `discover_history_levels`, `HISTORY_FIELD_NAMES`, `_HISTORY_COLUMN_PATTERN` (all in same module, from Tasks 4-5)
- Produces: `WideHistoryReshape` class (concrete `PreprocessingStrategy`). Used by `pipeline.py` (Task 19). Output columns: `PartID, WaferID, StageID, StepID, ToolID, ChuckID, Execute_Time, X_Posi, Y_Posi, NCE_Value, Pre_StageID, Pre_StepID, Pre_ToolID, Pre_ChamberID, Pre_Execute_Time`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wide_history_reshape.py`:
```python
from nce_analysis.preprocessing.wide_history_reshape import WideHistoryReshape


def _build_raw_df():
    return pd.DataFrame(
        [
            {
                "PartID": "P1",
                "WaferID": "W1",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": "LITHO_TOOL_A",
                "ChuckID": "CHK_1",
                "Execute_Time": "2026-01-01",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 20.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "CMP_STEP",
                "Pre_ToolID_1": "CMP_01",
                "Pre_ChamberID_1": "ChamberA",
                "Pre_Execute_Time_1": "2025-12-31",
                "Pre_StageID_2": "CVD",
                "Pre_StepID_2": "CVD_STEP",
                "Pre_ToolID_2": "CVD_01",
                "Pre_ChamberID_2": "ChamberX",
                "Pre_Execute_Time_2": "2025-12-30",
            },
            {
                "PartID": "P1",
                "WaferID": "W2",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": "LITHO_TOOL_A",
                "ChuckID": "CHK_1",
                "Execute_Time": "2026-01-02",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 5.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "CMP_STEP",
                "Pre_ToolID_1": "CMP_02",
                "Pre_ChamberID_1": "ChamberB",
                "Pre_Execute_Time_1": "2025-12-31",
                "Pre_StageID_2": None,
                "Pre_StepID_2": None,
                "Pre_ToolID_2": None,
                "Pre_ChamberID_2": None,
                "Pre_Execute_Time_2": None,
            },
        ]
    )


def test_transform_expands_history_levels_by_actual_stage_id():
    raw_df = _build_raw_df()

    result = WideHistoryReshape().transform(raw_df)

    assert len(result) == 3
    w1_stages = set(result[result["WaferID"] == "W1"]["Pre_StageID"])
    assert w1_stages == {"CMP", "CVD"}
    w2_rows = result[result["WaferID"] == "W2"]
    assert len(w2_rows) == 1
    assert w2_rows["Pre_StageID"].iloc[0] == "CMP"


def test_transform_raises_when_no_history_columns_present():
    raw_df = pd.DataFrame(
        [{"WaferID": "W1", "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 1.0}]}]
    )

    with pytest.raises(PreprocessingError):
        WideHistoryReshape().transform(raw_df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wide_history_reshape.py -v`
Expected: FAIL with `ImportError: cannot import name 'WideHistoryReshape'`

- [ ] **Step 3: Write minimal implementation**

Add to `nce_analysis/preprocessing/wide_history_reshape.py` (`PreprocessingError` and `pytest` are already imported at the top of the module/test file from Task 4):
```python
from nce_analysis.preprocessing.base import PreprocessingStrategy


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wide_history_reshape.py -v`
Expected: PASS (8 tests total in this file)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/preprocessing/wide_history_reshape.py tests/test_wide_history_reshape.py
git commit -m "feat: implement WideHistoryReshape preprocessing strategy"
```

---

## Task 7: Hotspot Strategy Base Interface

**Files:**
- Create: `nce_analysis/hotspot/__init__.py`
- Create: `nce_analysis/hotspot/base.py`
- Test: `tests/test_hotspot_base.py`

**Interfaces:**
- Consumes: `AnalysisConfig` from `nce_analysis.config`
- Produces: `HotspotStrategy` (ABC with abstract `detect(long_df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame`). Return DataFrame columns: `X_Posi, Y_Posi, anomalous_wafer_count, total_wafer_count, anomaly_ratio, insufficient_sample` (bool). Used by `ratio_threshold.py` (Task 8) and `pipeline.py` (Task 19).

- [ ] **Step 1: Write the failing test**

`tests/test_hotspot_base.py`:
```python
import pandas as pd
import pytest

from nce_analysis.config import AnalysisConfig
from nce_analysis.hotspot.base import HotspotStrategy


def test_hotspot_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        HotspotStrategy()


def test_concrete_subclass_can_implement_detect():
    class DummyStrategy(HotspotStrategy):
        def detect(self, long_df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
            return long_df

    strategy = DummyStrategy()
    df = pd.DataFrame({"a": [1, 2]})
    result = strategy.detect(df, AnalysisConfig())
    pd.testing.assert_frame_equal(result, df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hotspot_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.hotspot'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/hotspot/__init__.py`:
```python
```
(empty file — package marker)

`nce_analysis/hotspot/base.py`:
```python
from abc import ABC, abstractmethod

import pandas as pd

from nce_analysis.config import AnalysisConfig


class HotspotStrategy(ABC):
    @abstractmethod
    def detect(self, long_df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
        """long_df: standard long-format table (see preprocessing). Returns a
        DataFrame with columns X_Posi, Y_Posi, anomalous_wafer_count,
        total_wafer_count, anomaly_ratio, insufficient_sample (bool), sorted
        by anomaly_ratio desc, restricted to coordinates meeting
        config.hotspot_ratio_threshold. Coordinates below
        config.min_wafer_count are kept but flagged insufficient_sample=True:
        reported raw, excluded from root-cause analysis."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hotspot_base.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/hotspot/__init__.py nce_analysis/hotspot/base.py tests/test_hotspot_base.py
git commit -m "feat: add HotspotStrategy interface"
```

---

## Task 8: RatioThreshold Hotspot Detection

**Files:**
- Create: `nce_analysis/hotspot/ratio_threshold.py`
- Test: `tests/test_ratio_threshold.py`

**Interfaces:**
- Consumes: `HotspotStrategy` from `nce_analysis.hotspot.base`; `AnalysisConfig` from `nce_analysis.config`
- Produces: `RatioThreshold` class (concrete `HotspotStrategy`). Used by `pipeline.py` (Task 19).

- [ ] **Step 1: Write the failing test**

`tests/test_ratio_threshold.py`:
```python
import pandas as pd
import pytest

from nce_analysis.config import AnalysisConfig
from nce_analysis.hotspot.ratio_threshold import RatioThreshold


def _long_df():
    rows = []
    for i in range(6):
        rows.append(
            {
                "WaferID": f"W{i}",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0 if i < 4 else 5.0,
            }
        )
    for i in range(6):
        rows.append({"WaferID": f"W{i}", "X_Posi": 6.0, "Y_Posi": 0.0, "NCE_Value": 5.0})
    for i in range(2):
        rows.append({"WaferID": f"S{i}", "X_Posi": 12.0, "Y_Posi": 0.0, "NCE_Value": 20.0})
    return pd.DataFrame(rows)


def test_detect_identifies_hotspot_above_ratio_threshold():
    long_df = _long_df()
    config = AnalysisConfig(spec_threshold=15.0, min_wafer_count=5, hotspot_ratio_threshold=0.05)

    result = RatioThreshold().detect(long_df, config)

    sufficient = result[~result["insufficient_sample"]]
    assert len(sufficient) == 1
    row = sufficient.iloc[0]
    assert (row["X_Posi"], row["Y_Posi"]) == (0.0, 0.0)
    assert row["anomalous_wafer_count"] == 4
    assert row["total_wafer_count"] == 6
    assert row["anomaly_ratio"] == pytest.approx(4 / 6)


def test_detect_flags_coordinate_below_min_wafer_count():
    long_df = _long_df()
    config = AnalysisConfig(min_wafer_count=5)

    result = RatioThreshold().detect(long_df, config)

    flagged = result[(result["X_Posi"] == 12.0) & (result["Y_Posi"] == 0.0)]
    assert len(flagged) == 1
    assert bool(flagged["insufficient_sample"].iloc[0]) is True
    assert flagged["anomaly_ratio"].iloc[0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ratio_threshold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.hotspot.ratio_threshold'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/hotspot/ratio_threshold.py`:
```python
import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.hotspot.base import HotspotStrategy

LITHO_POINT_KEY_COLUMNS = ["WaferID", "X_Posi", "Y_Posi"]


class RatioThreshold(HotspotStrategy):
    def detect(self, long_df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
        litho_points = long_df.drop_duplicates(subset=LITHO_POINT_KEY_COLUMNS).copy()
        litho_points["is_anomaly"] = litho_points["NCE_Value"] > config.spec_threshold

        grouped = (
            litho_points.groupby(["X_Posi", "Y_Posi"])
            .agg(
                anomalous_wafer_count=("is_anomaly", "sum"),
                total_wafer_count=("is_anomaly", "count"),
            )
            .reset_index()
        )
        grouped["anomaly_ratio"] = (
            grouped["anomalous_wafer_count"] / grouped["total_wafer_count"]
        )

        grouped = grouped[grouped["anomaly_ratio"] >= config.hotspot_ratio_threshold]
        grouped["insufficient_sample"] = (
            grouped["total_wafer_count"] < config.min_wafer_count
        )

        return grouped.sort_values(
            ["anomaly_ratio", "anomalous_wafer_count"], ascending=False
        ).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ratio_threshold.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/hotspot/ratio_threshold.py tests/test_ratio_threshold.py
git commit -m "feat: implement RatioThreshold hotspot detection strategy"
```

---

## Task 9: Noise Filter Base Interface

**Files:**
- Create: `nce_analysis/noise_filter/__init__.py`
- Create: `nce_analysis/noise_filter/base.py`
- Test: `tests/test_noise_filter_base.py`

**Interfaces:**
- Consumes: `AnalysisConfig` from `nce_analysis.config`; `RootCauseDetail` from `nce_analysis.schema`
- Produces: `NoiseFilterResult` (dataclass: `surviving_hotspots: pd.DataFrame`, `litho_self_issues: list[RootCauseDetail]`), `NoiseFilterStrategy` (ABC with abstract `filter(long_df, hotspots, config) -> NoiseFilterResult`). Used by `majority_rule.py` (Task 10) and `pipeline.py` (Task 19).

- [ ] **Step 1: Write the failing test**

`tests/test_noise_filter_base.py`:
```python
import pandas as pd
import pytest

from nce_analysis.config import AnalysisConfig
from nce_analysis.noise_filter.base import NoiseFilterResult, NoiseFilterStrategy


def test_noise_filter_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        NoiseFilterStrategy()


def test_concrete_subclass_can_implement_filter():
    class DummyStrategy(NoiseFilterStrategy):
        def filter(self, long_df, hotspots, config) -> NoiseFilterResult:
            return NoiseFilterResult(surviving_hotspots=hotspots, litho_self_issues=[])

    strategy = DummyStrategy()
    hotspots = pd.DataFrame({"X_Posi": [0.0], "Y_Posi": [0.0]})
    result = strategy.filter(pd.DataFrame(), hotspots, AnalysisConfig())
    assert result.litho_self_issues == []
    pd.testing.assert_frame_equal(result.surviving_hotspots, hotspots)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_noise_filter_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.noise_filter'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/noise_filter/__init__.py`:
```python
```
(empty file — package marker)

`nce_analysis/noise_filter/base.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_noise_filter_base.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/noise_filter/__init__.py nce_analysis/noise_filter/base.py tests/test_noise_filter_base.py
git commit -m "feat: add NoiseFilterStrategy interface and NoiseFilterResult"
```

---

## Task 10: MajorityRule Noise Filter

**Files:**
- Create: `nce_analysis/noise_filter/majority_rule.py`
- Test: `tests/test_majority_rule.py`

**Interfaces:**
- Consumes: `NoiseFilterStrategy`, `NoiseFilterResult` from `nce_analysis.noise_filter.base`; `AnalysisConfig` from `nce_analysis.config`; `RootCauseDetail` from `nce_analysis.schema`
- Produces: `MajorityRule` class (concrete `NoiseFilterStrategy`). Used by `pipeline.py` (Task 19).

- [ ] **Step 1: Write the failing test**

`tests/test_majority_rule.py`:
```python
import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.noise_filter.majority_rule import MajorityRule


def _hotspots(coords):
    return pd.DataFrame(coords, columns=["X_Posi", "Y_Posi"])


def test_chuck_majority_single_stage_classified_as_chuck_issue():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": f"W{i}",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "StageID": "LITHO_M1",
            }
            for i in range(6)
        ]
    )
    hotspots = _hotspots([(0.0, 0.0)])
    config = AnalysisConfig(noise_filter_majority_threshold=0.5)

    result = MajorityRule().filter(long_df, hotspots, config)

    assert len(result.surviving_hotspots) == 0
    assert len(result.litho_self_issues) == 1
    assert result.litho_self_issues[0].Root_Cause_Type == "LITHO_CHUCK_ISSUE"
    assert result.litho_self_issues[0].Suspect_Pre_ChamberID == "CHK_1"


def test_chuck_majority_multi_stage_classified_as_contamination():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": f"W{i}",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "StageID": "LITHO_M1" if i % 2 == 0 else "LITHO_M2",
            }
            for i in range(6)
        ]
    )
    hotspots = _hotspots([(0.0, 0.0)])
    config = AnalysisConfig(noise_filter_majority_threshold=0.5)

    result = MajorityRule().filter(long_df, hotspots, config)

    assert result.litho_self_issues[0].Root_Cause_Type == "LITHO_CHUCK_CONTAMINATION"


def test_tool_majority_without_chuck_majority_classified_as_tool_issue():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": f"W{i}",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": "LITHO_A",
                "ChuckID": f"CHK_{i % 3}",
                "StageID": "LITHO_M1",
            }
            for i in range(6)
        ]
    )
    hotspots = _hotspots([(0.0, 0.0)])
    config = AnalysisConfig(noise_filter_majority_threshold=0.5)

    result = MajorityRule().filter(long_df, hotspots, config)

    assert len(result.surviving_hotspots) == 0
    assert result.litho_self_issues[0].Root_Cause_Type == "LITHO_TOOL_ISSUE"
    assert result.litho_self_issues[0].Suspect_Pre_ToolID == "LITHO_A"


def test_no_majority_survives_to_root_cause_analysis():
    long_df = pd.DataFrame(
        [
            {
                "WaferID": f"W{i}",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 20.0,
                "ToolID": f"LITHO_{i % 3}",
                "ChuckID": f"CHK_{i % 3}",
                "StageID": "LITHO_M1",
            }
            for i in range(6)
        ]
    )
    hotspots = _hotspots([(0.0, 0.0)])
    config = AnalysisConfig(noise_filter_majority_threshold=0.5)

    result = MajorityRule().filter(long_df, hotspots, config)

    assert len(result.surviving_hotspots) == 1
    assert len(result.litho_self_issues) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_majority_rule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.noise_filter.majority_rule'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/noise_filter/majority_rule.py`:
```python
import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.noise_filter.base import NoiseFilterResult, NoiseFilterStrategy
from nce_analysis.schema import RootCauseDetail


class MajorityRule(NoiseFilterStrategy):
    def filter(
        self, long_df: pd.DataFrame, hotspots: pd.DataFrame, config: AnalysisConfig
    ) -> NoiseFilterResult:
        litho_points = long_df.drop_duplicates(subset=["WaferID", "X_Posi", "Y_Posi"]).copy()
        litho_points["is_anomaly"] = litho_points["NCE_Value"] > config.spec_threshold

        surviving_rows = []
        self_issues: list[RootCauseDetail] = []

        for _, hotspot in hotspots.iterrows():
            coord_mask = (
                (litho_points["X_Posi"] == hotspot["X_Posi"])
                & (litho_points["Y_Posi"] == hotspot["Y_Posi"])
                & litho_points["is_anomaly"]
            )
            anomalous = litho_points[coord_mask]
            total_anomalous = len(anomalous)
            if total_anomalous == 0:
                surviving_rows.append(hotspot)
                continue

            chuck_counts = anomalous["ChuckID"].value_counts()
            top_chuck, chuck_n = chuck_counts.index[0], chuck_counts.iloc[0]
            chuck_share = chuck_n / total_anomalous

            if chuck_share > config.noise_filter_majority_threshold:
                chuck_rows = anomalous[anomalous["ChuckID"] == top_chuck]
                stage_diversity = chuck_rows["StageID"].nunique()
                owning_tool = chuck_rows["ToolID"].mode().iloc[0]
                root_cause_type = (
                    "LITHO_CHUCK_CONTAMINATION" if stage_diversity > 1 else "LITHO_CHUCK_ISSUE"
                )
                self_issues.append(
                    RootCauseDetail(
                        Suspect_Pre_ToolID=owning_tool,
                        Suspect_Pre_ChamberID=top_chuck,
                        Suspect_Pre_StepID="N/A",
                        Confidence_Score=chuck_share * 100,
                        Root_Cause_Type=root_cause_type,
                        Affected_Coordinates=[(hotspot["X_Posi"], hotspot["Y_Posi"])],
                        Metrics={
                            "chuck_share": chuck_share,
                            "stage_diversity": float(stage_diversity),
                        },
                    )
                )
                continue

            tool_counts = anomalous["ToolID"].value_counts()
            top_tool, tool_n = tool_counts.index[0], tool_counts.iloc[0]
            tool_share = tool_n / total_anomalous

            if tool_share > config.noise_filter_majority_threshold:
                self_issues.append(
                    RootCauseDetail(
                        Suspect_Pre_ToolID=top_tool,
                        Suspect_Pre_ChamberID="N/A",
                        Suspect_Pre_StepID="N/A",
                        Confidence_Score=tool_share * 100,
                        Root_Cause_Type="LITHO_TOOL_ISSUE",
                        Affected_Coordinates=[(hotspot["X_Posi"], hotspot["Y_Posi"])],
                        Metrics={"tool_share": tool_share},
                    )
                )
                continue

            surviving_rows.append(hotspot)

        surviving_df = (
            pd.DataFrame(surviving_rows) if surviving_rows else hotspots.iloc[0:0].copy()
        )
        return NoiseFilterResult(surviving_hotspots=surviving_df, litho_self_issues=self_issues)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_majority_rule.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/noise_filter/majority_rule.py tests/test_majority_rule.py
git commit -m "feat: implement MajorityRule noise filter strategy"
```

---

## Task 11: Root Cause Strategy Base Interface

**Files:**
- Create: `nce_analysis/root_cause/__init__.py`
- Create: `nce_analysis/root_cause/base.py`
- Test: `tests/test_root_cause_base.py`

**Interfaces:**
- Consumes: `AnalysisConfig` from `nce_analysis.config`
- Produces: `RootCauseCandidate` (dataclass: `suspect_tool_id: str`, `suspect_chamber_id: str`, `confidence_score: float`, `metrics: dict[str, float]`, `requires_manual_review: bool = False`), `RootCauseStrategy` (ABC with abstract `analyze(group_df, config) -> list[RootCauseCandidate]`). Used by `statistical.py` (Task 12), `ml.py` (Task 13), `both.py` (Task 14), `pipeline.py` (Task 19). Empty list means no suspect found; a list of 2 means sub-strategies disagreed (cross-validation case).

- [ ] **Step 1: Write the failing test**

`tests/test_root_cause_base.py`:
```python
import pandas as pd
import pytest

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import RootCauseCandidate, RootCauseStrategy


def test_root_cause_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        RootCauseStrategy()


def test_root_cause_candidate_defaults():
    candidate = RootCauseCandidate(
        suspect_tool_id="CMP_01", suspect_chamber_id="ChamberA", confidence_score=90.0
    )
    assert candidate.metrics == {}
    assert candidate.requires_manual_review is False


def test_concrete_subclass_can_implement_analyze():
    class DummyStrategy(RootCauseStrategy):
        def analyze(self, group_df: pd.DataFrame, config: AnalysisConfig):
            return []

    strategy = DummyStrategy()
    result = strategy.analyze(pd.DataFrame(), AnalysisConfig())
    assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_root_cause_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.root_cause'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/root_cause/__init__.py`:
```python
```
(empty file — package marker)

`nce_analysis/root_cause/base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from nce_analysis.config import AnalysisConfig


@dataclass
class RootCauseCandidate:
    suspect_tool_id: str
    suspect_chamber_id: str
    confidence_score: float
    metrics: dict[str, float] = field(default_factory=dict)
    requires_manual_review: bool = False


class RootCauseStrategy(ABC):
    @abstractmethod
    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        """group_df: rows for wafers at one hotspot coordinate and a single
        (Pre_StageID, Pre_StepID) group, with columns Pre_ToolID,
        Pre_ChamberID, is_anomaly.
        Returns a list of suspect Pre_ToolID+Pre_ChamberID candidates (usually
        0 or 1 entries; the 'both' cross-validation strategy may return 2 when
        its sub-strategies disagree, each marked requires_manual_review=True)."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_root_cause_base.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/root_cause/__init__.py nce_analysis/root_cause/base.py tests/test_root_cause_base.py
git commit -m "feat: add RootCauseStrategy interface and RootCauseCandidate"
```

---

## Task 12: Statistical Root Cause Strategy

**Files:**
- Create: `nce_analysis/root_cause/statistical.py`
- Test: `tests/test_statistical.py`

**Interfaces:**
- Consumes: `RootCauseStrategy`, `RootCauseCandidate` from `nce_analysis.root_cause.base`; `AnalysisConfig` from `nce_analysis.config`
- Produces: `StatisticalStrategy` class. Used by `both.py` (Task 14) and `pipeline.py` (Task 19).

- [ ] **Step 1: Write the failing test**

`tests/test_statistical.py`:
```python
import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.statistical import StatisticalStrategy


def _group_df_with_significant_combo():
    rows = []
    for i in range(10):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberA", "is_anomaly": i < 9})
    for i in range(10):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberB", "is_anomaly": i < 1})
    for i in range(10):
        rows.append({"Pre_ToolID": "CMP_02", "Pre_ChamberID": "ChamberC", "is_anomaly": i < 1})
    return pd.DataFrame(rows)


def test_analyze_identifies_significant_combo():
    group_df = _group_df_with_significant_combo()
    config = AnalysisConfig(alpha=0.05)

    result = StatisticalStrategy().analyze(group_df, config)

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_01"
    assert result[0].suspect_chamber_id == "ChamberA"
    assert result[0].confidence_score > 90


def test_analyze_returns_empty_when_no_significant_association():
    rows = []
    for combo_tool, combo_chamber in [("CMP_01", "ChamberA"), ("CMP_02", "ChamberB")]:
        for i in range(10):
            rows.append(
                {"Pre_ToolID": combo_tool, "Pre_ChamberID": combo_chamber, "is_anomaly": i < 5}
            )
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig(alpha=0.05)

    result = StatisticalStrategy().analyze(group_df, config)

    assert result == []


def test_analyze_uses_fisher_fallback_for_small_expected_counts():
    rows = []
    for _ in range(5):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberA", "is_anomaly": True})
    for _ in range(5):
        rows.append({"Pre_ToolID": "CMP_02", "Pre_ChamberID": "ChamberB", "is_anomaly": False})
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig(alpha=0.05)

    result = StatisticalStrategy().analyze(group_df, config)

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_01"
    assert result[0].metrics["fisher_fallback"] == 1.0
    assert "p_value_global" not in result[0].metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_statistical.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.root_cause.statistical'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/root_cause/statistical.py`:
```python
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats.contingency import expected_freq

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import RootCauseCandidate, RootCauseStrategy


class StatisticalStrategy(RootCauseStrategy):
    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        working = group_df.copy()
        working["combo"] = working["Pre_ToolID"] + "|" + working["Pre_ChamberID"]

        contingency = pd.crosstab(working["combo"], working["is_anomaly"])
        for col in (True, False):
            if col not in contingency.columns:
                contingency[col] = 0
        contingency = contingency[[True, False]]

        if contingency.shape[0] < 2:
            return []

        # Chi-square is unreliable when any expected cell count is < 5; in
        # that case skip the global gate and rely on the per-combo one-vs-rest
        # Fisher exact tests below (spec §7 fallback behavior).
        expected = expected_freq(contingency.to_numpy())
        fisher_fallback = bool((expected < 5).any())
        p_global = None
        if not fisher_fallback:
            _, p_global, _, _ = chi2_contingency(contingency)
            if p_global >= config.alpha:
                return []

        overall_anomaly_count = contingency[True].sum()
        overall_normal_count = contingency[False].sum()

        best_combo = None
        best_p = 1.0
        best_odds_ratio = 0.0
        for combo in contingency.index:
            combo_anomaly = contingency.loc[combo, True]
            combo_normal = contingency.loc[combo, False]
            rest_anomaly = overall_anomaly_count - combo_anomaly
            rest_normal = overall_normal_count - combo_normal
            table = [[combo_anomaly, combo_normal], [rest_anomaly, rest_normal]]
            odds_ratio, p_value = fisher_exact(table, alternative="greater")
            if p_value < best_p and odds_ratio > 1:
                best_p = p_value
                best_combo = combo
                best_odds_ratio = odds_ratio

        if best_combo is None or best_p >= config.alpha:
            return []

        tool_id, chamber_id = best_combo.split("|", 1)
        confidence_score = (1 - best_p) * 100
        metrics = {
            "p_value_combo": float(best_p),
            "odds_ratio": float(best_odds_ratio),
            "sample_size": float(len(working)),
            "fisher_fallback": 1.0 if fisher_fallback else 0.0,
        }
        if p_global is not None:
            metrics["p_value_global"] = float(p_global)
        return [
            RootCauseCandidate(
                suspect_tool_id=tool_id,
                suspect_chamber_id=chamber_id,
                confidence_score=confidence_score,
                metrics=metrics,
            )
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_statistical.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/root_cause/statistical.py tests/test_statistical.py
git commit -m "feat: implement StatisticalStrategy root cause analysis"
```

---

## Task 13: ML Root Cause Strategy

**Files:**
- Create: `nce_analysis/root_cause/ml.py`
- Test: `tests/test_ml.py`

**Interfaces:**
- Consumes: `RootCauseStrategy`, `RootCauseCandidate` from `nce_analysis.root_cause.base`; `AnalysisConfig` from `nce_analysis.config`
- Produces: `MLStrategy` class. Used by `both.py` (Task 14) and `pipeline.py` (Task 19).

- [ ] **Step 1: Write the failing test**

`tests/test_ml.py`:
```python
import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.ml import MLStrategy


def _group_df():
    rows = []
    for i in range(30):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberA", "is_anomaly": i < 27})
    for i in range(30):
        rows.append({"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberB", "is_anomaly": i < 3})
    return pd.DataFrame(rows)


def test_analyze_identifies_top_contributing_combo():
    group_df = _group_df()
    config = AnalysisConfig()

    result = MLStrategy().analyze(group_df, config)

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_01"
    assert result[0].suspect_chamber_id == "ChamberA"
    assert result[0].confidence_score > 0


def test_analyze_returns_empty_with_single_combo():
    group_df = pd.DataFrame(
        [
            {"Pre_ToolID": "CMP_01", "Pre_ChamberID": "ChamberA", "is_anomaly": i < 5}
            for i in range(10)
        ]
    )
    config = AnalysisConfig()

    result = MLStrategy().analyze(group_df, config)

    assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.root_cause.ml'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/root_cause/ml.py`:
```python
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import RootCauseCandidate, RootCauseStrategy


class MLStrategy(RootCauseStrategy):
    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        working = group_df.copy()
        working["combo"] = working["Pre_ToolID"] + "|" + working["Pre_ChamberID"]

        if working["combo"].nunique() < 2:
            return []

        encoder = OneHotEncoder(sparse_output=False)
        features = encoder.fit_transform(working[["combo"]])
        labels = working["is_anomaly"].astype(int).to_numpy()

        if labels.sum() == 0 or labels.sum() == len(labels):
            return []

        model = DecisionTreeClassifier(max_depth=3, random_state=0)
        model.fit(features, labels)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        # Binary classifiers: older shap returns list[class0_array, class1_array];
        # newer shap returns one ndarray with a trailing class axis.
        if isinstance(shap_values, list):
            positive_shap = shap_values[1]
        elif shap_values.ndim == 3:
            positive_shap = shap_values[..., 1]
        else:
            positive_shap = shap_values

        mean_contribution = positive_shap.mean(axis=0)
        combo_names = encoder.categories_[0]
        best_idx = int(np.argmax(mean_contribution))

        if mean_contribution[best_idx] <= 0:
            return []

        tool_id, chamber_id = combo_names[best_idx].split("|", 1)
        total_positive = mean_contribution[mean_contribution > 0].sum()
        confidence_score = (
            (mean_contribution[best_idx] / total_positive) * 100 if total_positive > 0 else 0.0
        )

        return [
            RootCauseCandidate(
                suspect_tool_id=tool_id,
                suspect_chamber_id=chamber_id,
                confidence_score=float(confidence_score),
                metrics={
                    "shap_contribution": float(mean_contribution[best_idx]),
                    "sample_size": float(len(working)),
                },
            )
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/root_cause/ml.py tests/test_ml.py
git commit -m "feat: implement MLStrategy root cause analysis"
```

---

## Task 14: Both (Cross-Validation) Root Cause Strategy

**Files:**
- Create: `nce_analysis/root_cause/both.py`
- Test: `tests/test_both.py`

**Interfaces:**
- Consumes: `RootCauseStrategy`, `RootCauseCandidate` from `nce_analysis.root_cause.base`; `StatisticalStrategy` from `nce_analysis.root_cause.statistical` (Task 12); `MLStrategy` from `nce_analysis.root_cause.ml` (Task 13); `AnalysisConfig` from `nce_analysis.config`
- Produces: `BothStrategy` class. Used by `pipeline.py` (Task 19).

- [ ] **Step 1: Write the failing test**

`tests/test_both.py`:
```python
from unittest.mock import MagicMock

import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import RootCauseCandidate
from nce_analysis.root_cause.both import BothStrategy


def test_both_returns_single_merged_candidate_when_strategies_agree():
    strategy = BothStrategy()
    strategy._statistical = MagicMock()
    strategy._ml = MagicMock()
    strategy._statistical.analyze.return_value = [
        RootCauseCandidate("CMP_01", "ChamberA", 80.0, {"p_value_combo": 0.01})
    ]
    strategy._ml.analyze.return_value = [
        RootCauseCandidate("CMP_01", "ChamberA", 70.0, {"shap_contribution": 0.3})
    ]

    result = strategy.analyze(pd.DataFrame(), AnalysisConfig())

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_01"
    assert result[0].confidence_score == 80.0
    assert result[0].requires_manual_review is False


def test_both_returns_both_candidates_when_strategies_disagree():
    strategy = BothStrategy()
    strategy._statistical = MagicMock()
    strategy._ml = MagicMock()
    strategy._statistical.analyze.return_value = [
        RootCauseCandidate("CMP_01", "ChamberA", 80.0, {})
    ]
    strategy._ml.analyze.return_value = [
        RootCauseCandidate("CMP_02", "ChamberC", 70.0, {})
    ]

    result = strategy.analyze(pd.DataFrame(), AnalysisConfig())

    assert len(result) == 2
    assert all(c.requires_manual_review for c in result)


def test_both_returns_empty_when_neither_strategy_finds_a_suspect():
    strategy = BothStrategy()
    strategy._statistical = MagicMock()
    strategy._ml = MagicMock()
    strategy._statistical.analyze.return_value = []
    strategy._ml.analyze.return_value = []

    result = strategy.analyze(pd.DataFrame(), AnalysisConfig())

    assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_both.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.root_cause.both'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/root_cause/both.py`:
```python
import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import RootCauseCandidate, RootCauseStrategy
from nce_analysis.root_cause.ml import MLStrategy
from nce_analysis.root_cause.statistical import StatisticalStrategy


class BothStrategy(RootCauseStrategy):
    def __init__(self):
        self._statistical = StatisticalStrategy()
        self._ml = MLStrategy()

    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        statistical_results = self._statistical.analyze(group_df, config)
        ml_results = self._ml.analyze(group_df, config)

        if not statistical_results and not ml_results:
            return []
        if not statistical_results:
            for candidate in ml_results:
                candidate.requires_manual_review = True
            return ml_results
        if not ml_results:
            for candidate in statistical_results:
                candidate.requires_manual_review = True
            return statistical_results

        stat_candidate = statistical_results[0]
        ml_candidate = ml_results[0]
        agree = (
            stat_candidate.suspect_tool_id == ml_candidate.suspect_tool_id
            and stat_candidate.suspect_chamber_id == ml_candidate.suspect_chamber_id
        )
        merged_metrics = {
            **{f"statistical_{k}": v for k, v in stat_candidate.metrics.items()},
            **{f"ml_{k}": v for k, v in ml_candidate.metrics.items()},
        }

        if agree:
            return [
                RootCauseCandidate(
                    suspect_tool_id=stat_candidate.suspect_tool_id,
                    suspect_chamber_id=stat_candidate.suspect_chamber_id,
                    confidence_score=max(
                        stat_candidate.confidence_score, ml_candidate.confidence_score
                    ),
                    metrics=merged_metrics,
                    requires_manual_review=False,
                )
            ]

        stat_candidate.requires_manual_review = True
        ml_candidate.requires_manual_review = True
        return [stat_candidate, ml_candidate]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_both.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/root_cause/both.py tests/test_both.py
git commit -m "feat: implement BothStrategy cross-validated root cause analysis"
```

---

## Task 15: Drift Strategy Base Interface

**Files:**
- Create: `nce_analysis/drift/__init__.py`
- Create: `nce_analysis/drift/base.py`
- Test: `tests/test_drift_base.py`

**Interfaces:**
- Consumes: `AnalysisConfig` from `nce_analysis.config`
- Produces: `DriftStrategy` (ABC with abstract `classify(series_df: pd.DataFrame, config: AnalysisConfig) -> tuple[str, dict[str, float]]`). Used by `regression_cusum.py` (Task 16), `correlation.py` (Task 17), `pipeline.py` (Task 19).

- [ ] **Step 1: Write the failing test**

`tests/test_drift_base.py`:
```python
import pandas as pd
import pytest

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.base import DriftStrategy


def test_drift_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        DriftStrategy()


def test_concrete_subclass_can_implement_classify():
    class DummyStrategy(DriftStrategy):
        def classify(self, series_df: pd.DataFrame, config: AnalysisConfig):
            return "SPECIFIC_CHAMBER_DEFECT", {}

    strategy = DummyStrategy()
    result = strategy.classify(pd.DataFrame(), AnalysisConfig())
    assert result == ("SPECIFIC_CHAMBER_DEFECT", {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.drift'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/drift/__init__.py`:
```python
```
(empty file — package marker)

`nce_analysis/drift/base.py`:
```python
from abc import ABC, abstractmethod

import pandas as pd

from nce_analysis.config import AnalysisConfig


class DriftStrategy(ABC):
    @abstractmethod
    def classify(
        self, series_df: pd.DataFrame, config: AnalysisConfig
    ) -> tuple[str, dict[str, float]]:
        """series_df: columns Pre_Execute_Time (datetime-parseable), NCE_Value
        for all wafers processed by one suspect Pre_ToolID+Pre_ChamberID
        combination within a single (Pre_StageID, Pre_StepID) group at the
        affected coordinate. Returns
        (root_cause_type, metrics) where root_cause_type is one of
        'SPECIFIC_CHAMBER_DEFECT', 'CHAMBER_DRIFT', 'CHAMBER_SUDDEN_SHIFT'."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_drift_base.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/drift/__init__.py nce_analysis/drift/base.py tests/test_drift_base.py
git commit -m "feat: add DriftStrategy interface"
```

---

## Task 16: RegressionCusum Drift Strategy

**Files:**
- Create: `nce_analysis/drift/regression_cusum.py`
- Test: `tests/test_regression_cusum.py`

**Interfaces:**
- Consumes: `DriftStrategy` from `nce_analysis.drift.base`; `AnalysisConfig` from `nce_analysis.config`
- Produces: `RegressionCusum` class. Used by `pipeline.py` (Task 19).

- [ ] **Step 1: Write the failing test**

`tests/test_regression_cusum.py`:
```python
import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.regression_cusum import RegressionCusum


def _timestamps(n):
    return pd.date_range("2026-01-01", periods=n, freq="h")


def test_classify_detects_gradual_drift():
    n = 20
    series_df = pd.DataFrame(
        {"Pre_Execute_Time": _timestamps(n), "NCE_Value": [10.0 + 0.5 * i for i in range(n)]}
    )
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, metrics = RegressionCusum().classify(series_df, config)

    assert root_cause_type == "CHAMBER_DRIFT"
    assert metrics["slope"] > 0


def test_classify_detects_sudden_shift():
    n = 20
    values = [10.0] * (n // 2) + [30.0] * (n // 2)
    series_df = pd.DataFrame({"Pre_Execute_Time": _timestamps(n), "NCE_Value": values})
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, _ = RegressionCusum().classify(series_df, config)

    assert root_cause_type == "CHAMBER_SUDDEN_SHIFT"


def test_classify_detects_stable_defect_with_no_trend():
    n = 21
    pattern = [20.0, 20.2, 19.8]
    values = (pattern * (n // 3 + 1))[:n]
    series_df = pd.DataFrame({"Pre_Execute_Time": _timestamps(n), "NCE_Value": values})
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, _ = RegressionCusum().classify(series_df, config)

    assert root_cause_type == "SPECIFIC_CHAMBER_DEFECT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_regression_cusum.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.drift.regression_cusum'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/drift/regression_cusum.py`:
```python
import numpy as np
import pandas as pd
from scipy import stats

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.base import DriftStrategy


class RegressionCusum(DriftStrategy):
    def classify(
        self, series_df: pd.DataFrame, config: AnalysisConfig
    ) -> tuple[str, dict[str, float]]:
        ordered = series_df.sort_values("Pre_Execute_Time").reset_index(drop=True)
        timestamps = pd.to_datetime(ordered["Pre_Execute_Time"])
        elapsed_hours = (
            (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
        ).to_numpy()
        values = ordered["NCE_Value"].to_numpy()

        slope, intercept, _, p_value, _ = stats.linregress(elapsed_hours, values)
        fitted = intercept + slope * elapsed_hours
        residuals = values - fitted

        cusum = np.cumsum(residuals - residuals.mean())
        cusum_range = float(cusum.max() - cusum.min())
        residual_std = float(residuals.std())
        cusum_threshold = 5 * residual_std if residual_std > 0 else 0.0
        change_point_detected = cusum_threshold > 0 and cusum_range > cusum_threshold

        metrics = {
            "slope": float(slope),
            "slope_p_value": float(p_value),
            "cusum_range": cusum_range,
            "cusum_threshold": cusum_threshold,
        }

        if change_point_detected:
            return "CHAMBER_SUDDEN_SHIFT", metrics
        if p_value < config.alpha and slope > 0:
            return "CHAMBER_DRIFT", metrics
        return "SPECIFIC_CHAMBER_DEFECT", metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_regression_cusum.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/drift/regression_cusum.py tests/test_regression_cusum.py
git commit -m "feat: implement RegressionCusum drift strategy"
```

---

## Task 17: Correlation Drift Strategy

**Files:**
- Create: `nce_analysis/drift/correlation.py`
- Test: `tests/test_correlation.py`

**Interfaces:**
- Consumes: `DriftStrategy` from `nce_analysis.drift.base`; `AnalysisConfig` from `nce_analysis.config`
- Produces: `Correlation` class. Used by `pipeline.py` (Task 19).

- [ ] **Step 1: Write the failing test**

`tests/test_correlation.py`:
```python
import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.correlation import Correlation


def _timestamps(n):
    return pd.date_range("2026-01-01", periods=n, freq="h")


def test_classify_detects_drift_via_correlation():
    n = 20
    series_df = pd.DataFrame(
        {"Pre_Execute_Time": _timestamps(n), "NCE_Value": [10.0 + 0.5 * i for i in range(n)]}
    )
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, metrics = Correlation().classify(series_df, config)

    assert root_cause_type == "CHAMBER_DRIFT"
    assert metrics["correlation"] > 0.9


def test_classify_returns_stable_when_uncorrelated():
    n = 21
    pattern = [20.0, 20.2, 19.8]
    values = (pattern * (n // 3 + 1))[:n]
    series_df = pd.DataFrame({"Pre_Execute_Time": _timestamps(n), "NCE_Value": values})
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, _ = Correlation().classify(series_df, config)

    assert root_cause_type == "SPECIFIC_CHAMBER_DEFECT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_correlation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.drift.correlation'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/drift/correlation.py`:
```python
import pandas as pd
from scipy import stats

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.base import DriftStrategy


class Correlation(DriftStrategy):
    def classify(
        self, series_df: pd.DataFrame, config: AnalysisConfig
    ) -> tuple[str, dict[str, float]]:
        ordered = series_df.sort_values("Pre_Execute_Time").reset_index(drop=True)
        timestamps = pd.to_datetime(ordered["Pre_Execute_Time"])
        elapsed_hours = (
            (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
        ).to_numpy()
        values = ordered["NCE_Value"].to_numpy()

        correlation, p_value = stats.pearsonr(elapsed_hours, values)
        metrics = {"correlation": float(correlation), "p_value": float(p_value)}

        if p_value < config.alpha and correlation > 0:
            return "CHAMBER_DRIFT", metrics
        return "SPECIFIC_CHAMBER_DEFECT", metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_correlation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/drift/correlation.py tests/test_correlation.py
git commit -m "feat: implement Correlation drift strategy"
```

---

## Task 18: Result Aggregator

**Files:**
- Create: `nce_analysis/result.py`
- Test: `tests/test_result.py`

**Interfaces:**
- Consumes: `AnalysisConfig` from `nce_analysis.config`; `RootCauseDetail`, `AnalysisResult`, `InsufficientSampleHotspot` from `nce_analysis.schema`
- Produces: `aggregate_results(details: list[RootCauseDetail], config: AnalysisConfig, insufficient_sample_hotspots: list[InsufficientSampleHotspot] | None = None) -> AnalysisResult`. `Details` keeps per-coordinate granularity sorted by confidence desc. `Summary` merges details sharing `(Suspect_Pre_ToolID, Suspect_Pre_ChamberID, Suspect_Pre_StepID, Root_Cause_Type)` into one entry (coordinates unioned, confidence/metrics from the highest-confidence member, `Requires_Manual_Review` OR-ed) before truncating to `summary_top_n` — so top-N is not filled by one suspect repeated per coordinate. Used by `pipeline.py` (Task 19).

- [ ] **Step 1: Write the failing test**

`tests/test_result.py`:
```python
from nce_analysis.config import AnalysisConfig
from nce_analysis.result import aggregate_results
from nce_analysis.schema import InsufficientSampleHotspot, RootCauseDetail


def _detail(score, coord=(0.0, 0.0), chamber="ChamberA"):
    return RootCauseDetail(
        Suspect_Pre_ToolID="CMP_01",
        Suspect_Pre_ChamberID=chamber,
        Suspect_Pre_StepID="3580.01",
        Confidence_Score=score,
        Root_Cause_Type="SPECIFIC_CHAMBER_DEFECT",
        Affected_Coordinates=[coord],
        Metrics={},
    )


def test_aggregate_results_sorts_details_by_confidence_descending():
    details = [_detail(50.0), _detail(90.0), _detail(70.0)]
    config = AnalysisConfig(summary_top_n=2)

    result = aggregate_results(details, config)

    assert [d.Confidence_Score for d in result.Details] == [90.0, 70.0, 50.0]
    assert result.Config_Used.summary_top_n == 2


def test_aggregate_results_merges_same_suspect_into_one_summary_entry():
    details = [
        _detail(90.0, coord=(0.0, 0.0)),
        _detail(70.0, coord=(6.0, 0.0)),
        _detail(60.0, coord=(12.0, 0.0), chamber="ChamberB"),
    ]
    config = AnalysisConfig(summary_top_n=5)

    result = aggregate_results(details, config)

    assert len(result.Summary) == 2
    top = result.Summary[0]
    assert top.Suspect_Pre_ChamberID == "ChamberA"
    assert top.Confidence_Score == 90.0
    assert set(top.Affected_Coordinates) == {(0.0, 0.0), (6.0, 0.0)}
    assert result.Summary[1].Suspect_Pre_ChamberID == "ChamberB"
    assert len(result.Details) == 3


def test_aggregate_results_handles_empty_details_and_carries_insufficient():
    hotspot = InsufficientSampleHotspot(
        X_Posi=0.0,
        Y_Posi=0.0,
        anomaly_ratio=1.0,
        anomalous_wafer_count=2,
        total_wafer_count=2,
    )

    result = aggregate_results([], AnalysisConfig(), [hotspot])

    assert result.Summary == []
    assert result.Details == []
    assert result.Insufficient_Sample_Hotspots == [hotspot]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_result.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.result'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/result.py`:
```python
from datetime import datetime, timezone

from nce_analysis.config import AnalysisConfig
from nce_analysis.schema import AnalysisResult, InsufficientSampleHotspot, RootCauseDetail


def aggregate_results(
    details: list[RootCauseDetail],
    config: AnalysisConfig,
    insufficient_sample_hotspots: list[InsufficientSampleHotspot] | None = None,
) -> AnalysisResult:
    sorted_details = sorted(details, key=lambda d: d.Confidence_Score, reverse=True)

    # Summary: one entry per suspect. Iterating in confidence order means the
    # first detail seen per key carries the group's max confidence and metrics,
    # and dict insertion order keeps the summary confidence-sorted.
    merged: dict[tuple, RootCauseDetail] = {}
    for detail in sorted_details:
        key = (
            detail.Suspect_Pre_ToolID,
            detail.Suspect_Pre_ChamberID,
            detail.Suspect_Pre_StepID,
            detail.Root_Cause_Type,
        )
        if key not in merged:
            merged[key] = detail.model_copy(deep=True)
            continue
        entry = merged[key]
        entry.Affected_Coordinates = entry.Affected_Coordinates + [
            coord
            for coord in detail.Affected_Coordinates
            if coord not in entry.Affected_Coordinates
        ]
        entry.Requires_Manual_Review = (
            entry.Requires_Manual_Review or detail.Requires_Manual_Review
        )

    return AnalysisResult(
        Summary=list(merged.values())[: config.summary_top_n],
        Details=sorted_details,
        Insufficient_Sample_Hotspots=insufficient_sample_hotspots or [],
        Generated_At=datetime.now(timezone.utc),
        Config_Used=config,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_result.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/result.py tests/test_result.py
git commit -m "feat: implement result aggregation"
```

---

## Task 19: Pipeline Orchestrator + End-to-End Test

**Files:**
- Create: `nce_analysis/pipeline.py`
- Test: `tests/test_pipeline_e2e.py`

**Interfaces:**
- Consumes: `AnalysisConfig` (Task 1), `AnalysisResult`, `RootCauseDetail`, `InsufficientSampleHotspot` (Task 2), `WideHistoryReshape` (Task 6), `RatioThreshold` (Task 8), `MajorityRule` (Task 10), `StatisticalStrategy` (Task 12), `MLStrategy` (Task 13), `BothStrategy` (Task 14), `RegressionCusum` (Task 16), `Correlation` (Task 17), `aggregate_results` (Task 18)
- Produces: `run(raw_df: pd.DataFrame, config: AnalysisConfig | None = None) -> AnalysisResult` — the library's single public entry point.

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline_e2e.py`:
```python
import pandas as pd

from nce_analysis import pipeline
from nce_analysis.config import AnalysisConfig


def _build_batch():
    rows = []
    litho_tools = ["LITHO_A", "LITHO_B", "LITHO_C"]
    litho_chucks = ["CHK_1", "CHK_2", "CHK_3"]

    for i in range(18):
        rows.append(
            {
                "PartID": "PART1",
                "WaferID": f"ANOM_{i}",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": litho_tools[i % 3],
                "ChuckID": litho_chucks[i % 3],
                "Execute_Time": f"2026-01-{(i % 28) + 1:02d}",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 25.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "CMP_STEP",
                "Pre_ToolID_1": "CMP_01",
                "Pre_ChamberID_1": "ChamberA",
                "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
            }
        )
    for i in range(6):
        rows.append(
            {
                "PartID": "PART1",
                "WaferID": f"NORM_B_{i}",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": litho_tools[i % 3],
                "ChuckID": litho_chucks[i % 3],
                "Execute_Time": f"2026-02-{(i % 28) + 1:02d}",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 5.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "CMP_STEP",
                "Pre_ToolID_1": "CMP_01",
                "Pre_ChamberID_1": "ChamberB",
                "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
            }
        )
    for i in range(6):
        rows.append(
            {
                "PartID": "PART1",
                "WaferID": f"NORM_C_{i}",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": litho_tools[i % 3],
                "ChuckID": litho_chucks[i % 3],
                "Execute_Time": f"2026-03-{(i % 28) + 1:02d}",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 5.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "CMP_STEP",
                "Pre_ToolID_1": "CMP_02",
                "Pre_ChamberID_1": "ChamberC",
                "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
            }
        )
    return pd.DataFrame(rows)


def test_pipeline_identifies_injected_root_cause():
    raw_df = _build_batch()
    config = AnalysisConfig(root_cause_strategy="statistical", drift_strategy="regression_cusum")

    result = pipeline.run(raw_df, config)

    assert len(result.Details) >= 1
    top = result.Summary[0]
    assert top.Suspect_Pre_ToolID == "CMP_01"
    assert top.Suspect_Pre_ChamberID == "ChamberA"
    assert top.Suspect_Pre_StepID == "CMP_STEP"
    assert top.Root_Cause_Type in {
        "SPECIFIC_CHAMBER_DEFECT",
        "CHAMBER_DRIFT",
        "CHAMBER_SUDDEN_SHIFT",
    }
    assert top.Requires_Manual_Review is False


def test_pipeline_returns_empty_result_when_no_hotspots():
    raw_df = pd.DataFrame(
        [
            {
                "PartID": "P1",
                "WaferID": "W1",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "Execute_Time": "2026-01-01",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 5.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "CMP_STEP",
                "Pre_ToolID_1": "CMP_01",
                "Pre_ChamberID_1": "ChamberA",
                "Pre_Execute_Time_1": "2025-12-01",
            }
        ]
    )
    config = AnalysisConfig(min_wafer_count=5)

    result = pipeline.run(raw_df, config)

    assert result.Details == []
    assert result.Summary == []
    assert result.Insufficient_Sample_Hotspots == []


def test_pipeline_distinguishes_repeated_stage_visits_by_step_id():
    rows = []
    litho_tools = ["LITHO_A", "LITHO_B", "LITHO_C"]
    litho_chucks = ["CHK_1", "CHK_2", "CHK_3"]
    for i in range(24):
        anomalous = i < 12
        rows.append(
            {
                "PartID": "PART1",
                "WaferID": f"W{i}",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": litho_tools[i % 3],
                "ChuckID": litho_chucks[i % 3],
                "Execute_Time": f"2026-01-{(i % 28) + 1:02d}",
                "Measurement_Points": [
                    {"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 25.0 if anomalous else 5.0}
                ],
                # Second CMP pass (most recent history level): tool assignment
                # balanced across anomaly status — must NOT be blamed.
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "3580.02",
                "Pre_ToolID_1": "CMP_C" if i % 2 == 0 else "CMP_D",
                "Pre_ChamberID_1": "ChamberM" if i % 2 == 0 else "ChamberN",
                "Pre_Execute_Time_1": f"2025-12-{(i % 28) + 1:02d}",
                # First CMP pass (earlier history level): the injected culprit.
                "Pre_StageID_2": "CMP",
                "Pre_StepID_2": "3580.01",
                "Pre_ToolID_2": "CMP_A" if anomalous else "CMP_B",
                "Pre_ChamberID_2": "ChamberX" if anomalous else "ChamberY",
                "Pre_Execute_Time_2": f"2025-11-{(i % 28) + 1:02d}",
            }
        )
    raw_df = pd.DataFrame(rows)
    config = AnalysisConfig(root_cause_strategy="statistical")

    result = pipeline.run(raw_df, config)

    top = result.Summary[0]
    assert top.Suspect_Pre_ToolID == "CMP_A"
    assert top.Suspect_Pre_ChamberID == "ChamberX"
    assert top.Suspect_Pre_StepID == "3580.01"
    assert not any(d.Suspect_Pre_StepID == "3580.02" for d in result.Details)


def test_pipeline_reports_insufficient_sample_hotspot_without_confidence():
    rows = []
    for i in range(2):
        rows.append(
            {
                "PartID": "P1",
                "WaferID": f"W{i}",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "Execute_Time": f"2026-01-0{i + 1}",
                "Measurement_Points": [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 25.0}],
                "Pre_StageID_1": "CMP",
                "Pre_StepID_1": "3580.01",
                "Pre_ToolID_1": "CMP_01",
                "Pre_ChamberID_1": "ChamberA",
                "Pre_Execute_Time_1": f"2025-12-0{i + 1}",
            }
        )
    raw_df = pd.DataFrame(rows)
    config = AnalysisConfig(min_wafer_count=5)

    result = pipeline.run(raw_df, config)

    assert result.Details == []
    assert result.Summary == []
    assert len(result.Insufficient_Sample_Hotspots) == 1
    hotspot = result.Insufficient_Sample_Hotspots[0]
    assert (hotspot.X_Posi, hotspot.Y_Posi) == (0.0, 0.0)
    assert hotspot.anomaly_ratio == 1.0
    assert hotspot.total_wafer_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_e2e.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.pipeline'`

- [ ] **Step 3: Write minimal implementation**

`nce_analysis/pipeline.py`:
```python
import logging

import pandas as pd

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.correlation import Correlation
from nce_analysis.drift.regression_cusum import RegressionCusum
from nce_analysis.hotspot.ratio_threshold import RatioThreshold
from nce_analysis.noise_filter.majority_rule import MajorityRule
from nce_analysis.preprocessing.wide_history_reshape import WideHistoryReshape
from nce_analysis.result import aggregate_results
from nce_analysis.root_cause.both import BothStrategy
from nce_analysis.root_cause.ml import MLStrategy
from nce_analysis.root_cause.statistical import StatisticalStrategy
from nce_analysis.schema import AnalysisResult, InsufficientSampleHotspot, RootCauseDetail

logger = logging.getLogger(__name__)

_ROOT_CAUSE_STRATEGIES = {
    "statistical": StatisticalStrategy,
    "ml": MLStrategy,
    "both": BothStrategy,
}
_DRIFT_STRATEGIES = {
    "regression_cusum": RegressionCusum,
    "correlation": Correlation,
}


def run(raw_df: pd.DataFrame, config: AnalysisConfig | None = None) -> AnalysisResult:
    config = config or AnalysisConfig()

    preprocessor = WideHistoryReshape()
    hotspot_detector = RatioThreshold()
    noise_filter = MajorityRule()
    root_cause_strategy = _ROOT_CAUSE_STRATEGIES[config.root_cause_strategy]()
    drift_strategy = _DRIFT_STRATEGIES[config.drift_strategy]()

    long_df = preprocessor.transform(raw_df)
    hotspots = hotspot_detector.detect(long_df, config)

    if hotspots.empty:
        return aggregate_results([], config)

    insufficient_hotspots = [
        InsufficientSampleHotspot(
            X_Posi=row["X_Posi"],
            Y_Posi=row["Y_Posi"],
            anomaly_ratio=row["anomaly_ratio"],
            anomalous_wafer_count=int(row["anomalous_wafer_count"]),
            total_wafer_count=int(row["total_wafer_count"]),
        )
        for _, row in hotspots[hotspots["insufficient_sample"]].iterrows()
    ]
    hotspots = hotspots[~hotspots["insufficient_sample"]]

    if hotspots.empty:
        return aggregate_results([], config, insufficient_hotspots)

    noise_result = noise_filter.filter(long_df, hotspots, config)
    details: list[RootCauseDetail] = list(noise_result.litho_self_issues)

    for _, hotspot in noise_result.surviving_hotspots.iterrows():
        coord = (hotspot["X_Posi"], hotspot["Y_Posi"])
        point_rows = long_df[
            (long_df["X_Posi"] == coord[0]) & (long_df["Y_Posi"] == coord[1])
        ]

        # fillna keeps rows from a malformed history level (StageID present but
        # no StepID column) visible instead of silently vanishing from groupby.
        point_rows = point_rows.assign(
            Pre_StepID=point_rows["Pre_StepID"].fillna("UNKNOWN")
        )

        for (pre_stage_id, pre_step_id), step_group in point_rows.groupby(
            ["Pre_StageID", "Pre_StepID"]
        ):
            litho_points = (
                step_group.sort_values("Pre_Execute_Time")
                .drop_duplicates(subset=["WaferID"], keep="last")
                .copy()
            )
            if len(litho_points) < len(step_group):
                logger.warning(
                    "Duplicate wafer records for stage %s step %s at %s "
                    "(rework?); keeping the most recent Pre_Execute_Time per wafer.",
                    pre_stage_id,
                    pre_step_id,
                    coord,
                )
            litho_points["is_anomaly"] = (
                litho_points["NCE_Value"] > config.spec_threshold
            )

            candidates = root_cause_strategy.analyze(litho_points, config)

            for candidate in candidates:
                combo_history = litho_points[
                    (litho_points["Pre_ToolID"] == candidate.suspect_tool_id)
                    & (litho_points["Pre_ChamberID"] == candidate.suspect_chamber_id)
                ]

                if len(combo_history) < 3:
                    root_cause_type = "SPECIFIC_CHAMBER_DEFECT"
                    drift_metrics = {"insufficient_drift_sample": 1.0}
                else:
                    root_cause_type, drift_metrics = drift_strategy.classify(
                        combo_history[["Pre_Execute_Time", "NCE_Value"]], config
                    )

                details.append(
                    RootCauseDetail(
                        Suspect_Pre_ToolID=candidate.suspect_tool_id,
                        Suspect_Pre_ChamberID=candidate.suspect_chamber_id,
                        Suspect_Pre_StepID=pre_step_id,
                        Confidence_Score=candidate.confidence_score,
                        Root_Cause_Type=root_cause_type,
                        Affected_Coordinates=[coord],
                        Metrics={**candidate.metrics, **drift_metrics},
                        Requires_Manual_Review=candidate.requires_manual_review,
                    )
                )

    return aggregate_results(details, config, insufficient_hotspots)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_e2e.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `pytest -v`
Expected: PASS (all tests across all 19 tasks)

- [ ] **Step 6: Commit**

```bash
git add nce_analysis/pipeline.py tests/test_pipeline_e2e.py
git commit -m "feat: implement pipeline orchestrator and end-to-end test"
```

---

## Self-Review Notes

**Spec coverage:** §1 (pure library) → Task 1 constraint. §2 (input shape) → Tasks 4-6. §3 (architecture) → all strategy base/impl pairs. §4 (preprocessing) → Tasks 4-6. §5 (hotspot) → Tasks 7-8. §6 (noise filter) → Tasks 9-10. §7 (root cause: statistical/ml/both) → Tasks 11-14. §8 (drift) → Tasks 15-17. §9 (taxonomy, v1 scope exclusion of PRODUCT_TOOL_MISMATCH) → Task 19 (no PartID feature added, confirmed absent). §10 (schemas) → Tasks 1-2. §11 (error handling: PreprocessingError on missing history columns → Task 6; empty `Measurement_Points` skip/all-empty abort → Task 4; insufficient_sample reporting → Tasks 2, 8, 18, 19; chi-square→Fisher fallback → Task 12; rework de-dup with warning → Task 19). §12 (testing plan: per-strategy unit tests, noise-filter domain rules, preprocessing dynamic-N tests, e2e fixture) → covered throughout, e2e in Task 19.

**Type consistency verified:** `RootCauseStrategy.analyze` returns `list[RootCauseCandidate]` consistently across base (Task 11), statistical (Task 12), ml (Task 13), both (Task 14), and is consumed as a list in pipeline.py (Task 19). `NoiseFilterStrategy.filter` returns `NoiseFilterResult` consistently between base (Task 9), majority_rule (Task 10), and pipeline.py (Task 19). `DriftStrategy.classify` returns `tuple[str, dict[str, float]]` consistently across base (Task 15), regression_cusum (Task 16), correlation (Task 17), pipeline.py (Task 19).

**No placeholders:** every step contains complete, runnable code; no TBD/TODO markers.

**Review fixes incorporated:** wafers with empty `Measurement_Points` are skipped with a logged warning, and an all-empty batch raises `PreprocessingError` (Task 4). The statistical strategy gates the global chi-square on expected cell counts ≥ 5 and otherwise falls back to the one-vs-rest Fisher exact tests, surfaced via the `fisher_fallback` metric; the winning combo must itself satisfy p < alpha (Task 12). Below-`min_wafer_count` hotspots are flagged by `RatioThreshold` (Task 8) and reported raw in `AnalysisResult.Insufficient_Sample_Hotspots` (Tasks 2, 18, 19) — never given a confidence score. `Summary` merges per-coordinate details per suspect so top-N is not one chamber repeated per coordinate (Task 18). The ML strategy handles both list and 3-D ndarray SHAP return shapes across shap versions (Task 13). `MajorityRule` guards the zero-anomalous-wafer edge (Task 10). The unused `xgboost` dependency was removed (Task 1).

**Repeated stage visits:** root-cause grouping uses the composite `(Pre_StageID, Pre_StepID)` key in `pipeline.py` (Task 19), so a wafer that passed the same stage type twice (e.g. CMP steps `3580.01` and `3580.02`) contributes exactly one record to each step's analysis group instead of being arbitrarily de-duplicated. Residual duplicates within one group (rework/data duplication) keep the most recent `Pre_Execute_Time` record with a logged warning. `RootCauseDetail.Suspect_Pre_StepID` (Task 2) records which pass is blamed (`"N/A"` for LITHO self-issues from the noise filter, Task 10). Verified end-to-end by `test_pipeline_distinguishes_repeated_stage_visits_by_step_id` (Task 19).
