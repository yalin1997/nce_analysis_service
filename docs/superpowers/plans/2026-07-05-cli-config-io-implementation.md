# File-Based Input, YAML Config, and CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `nce_analysis` be driven from files instead of only in-process Python: load a wafer batch from `.csv`/`.parquet`, load an `AnalysisConfig` from YAML, and run the whole pipeline via `python -m nce_analysis --input ... [--config ...] [--output ...]`.

**Architecture:** Three additive pieces sit in front of the existing `pipeline.run(raw_df, config)` (unchanged): `nce_analysis/io.py::load_input` turns a CSV or Parquet file into the same `raw_df` shape the pipeline already expects; `AnalysisConfig.from_yaml` turns a YAML file into an `AnalysisConfig`; `nce_analysis/cli.py::main` wires both into `pipeline.run` and writes/prints the JSON result. `examples/` gains committed CSV/Parquet/YAML fixtures generated from the existing synthetic sample batch.

**Tech Stack:** Python 3.12, pandas (CSV/Parquet I/O), pyarrow (parquet engine), pyyaml (config parsing), pydantic v2 (`ConfigDict(extra="forbid")`, `model_validate`), argparse (stdlib CLI), pytest.

## Global Constraints

- Use `python3.12` explicitly for every command — no bare `python3`, no committed venv (per `CLAUDE.md`).
- Follow red/green TDD: write/update the failing test first, confirm the failure, then implement (per `CLAUDE.md`).
- `pipeline.run(raw_df, config)`'s signature and behavior do not change — every new piece sits in front of it (per spec §1).
- CLI is `python -m nce_analysis` only — no `[project.scripts]` console-script entry, and `nce_analysis/cli.py` itself has no `__main__` guard (per spec §5).
- No CLI flags to override individual `AnalysisConfig` fields — config is YAML-only (per spec §2).
- CSV supports two auto-detected shapes (wide: `Measurement_Points` JSON column; long: `X_Posi`/`Y_Posi`/`NCE_Value` columns + `WaferID`); wide takes priority when both could match. Parquet only supports the wide/native shape (per spec §3).
- After `pd.read_parquet`, `Measurement_Points` cells come back as `numpy.ndarray`, not `list` — must be normalized with `.map(list)` before the DataFrame reaches `preprocessing/`, or every wafer is silently treated as empty (per spec §3, verified empirically).
- `AnalysisConfig` gets `model_config = ConfigDict(extra="forbid")`, applied model-wide (not just to YAML loading) — verified no existing call site relies on the current silent-ignore-unknown-field behavior (per spec §4).
- Only three failure modes are caught and turned into a clean `stderr` message + exit code 1 in the CLI: `InputFormatError`/`FileNotFoundError` (input file), `FileNotFoundError`/`ValidationError` (config file), `PreprocessingError` (pipeline run). Everything else propagates as a normal traceback (per spec §5, §7).
- New dependencies: `pyyaml`, `pyarrow` (per spec §8).

---

## Task 1: `AnalysisConfig.from_yaml` + `extra="forbid"`

**Files:**
- Modify: `pyproject.toml`
- Modify: `nce_analysis/config.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `AnalysisConfig.from_yaml(path: str | Path) -> AnalysisConfig` (classmethod). `AnalysisConfig(...)` now raises `pydantic.ValidationError` for any unrecognized keyword argument. Later tasks (`cli.py`) call `AnalysisConfig.from_yaml(args.config)`.

- [ ] **Step 1: Add the `pyyaml` dependency**

Edit `pyproject.toml` — in the `dependencies` list, add `"pyyaml>=6.0",` as the last entry (after `"numpy>=1.24",`):

```toml
dependencies = [
    "pandas>=2.0",
    "pydantic>=2.0",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "shap>=0.44",
    "numpy>=1.24",
    "pyyaml>=6.0",
]
```

Run:
```bash
python3.12 -m pip install --user --break-system-packages -e ".[dev]"
```
Expected: install succeeds, pulling in `pyyaml`.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_config.py` (existing imports `pytest`, `ValidationError`, and `AnalysisConfig` are already present at the top of the file — do not duplicate them):

```python
def test_config_rejects_unknown_field():
    with pytest.raises(ValidationError):
        AnalysisConfig(typo_field=1)


def test_from_yaml_overrides_only_specified_fields(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("summary_top_n: 3\nspec_threshold: 20.0\n")

    config = AnalysisConfig.from_yaml(yaml_path)

    assert config.summary_top_n == 3
    assert config.spec_threshold == 20.0
    assert config.min_wafer_count == 5  # untouched field keeps its default


def test_from_yaml_empty_file_uses_all_defaults(tmp_path):
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("")

    config = AnalysisConfig.from_yaml(yaml_path)

    assert config == AnalysisConfig()


def test_from_yaml_missing_file_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        AnalysisConfig.from_yaml(missing_path)


def test_from_yaml_unknown_field_raises_validation_error(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("typo_field: 1\n")

    with pytest.raises(ValidationError):
        AnalysisConfig.from_yaml(yaml_path)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3.12 -m pytest tests/test_config.py -v`
Expected: the 5 new tests FAIL — `test_config_rejects_unknown_field` fails because extra fields are currently silently ignored (no error raised); the 4 `from_yaml` tests fail with `AttributeError: type object 'AnalysisConfig' has no attribute 'from_yaml'`.

- [ ] **Step 4: Implement**

Replace the full contents of `nce_analysis/config.py` with:

```python
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_threshold: float = 15.0
    min_wafer_count: int = Field(default=5, ge=1)
    hotspot_ratio_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    noise_filter_majority_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    root_cause_strategy: Literal["statistical", "ml", "both"] = "both"
    root_cause_granularity: Literal["tool", "chamber"] = "chamber"
    drift_strategy: Literal["regression_cusum", "correlation"] = "regression_cusum"
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    summary_top_n: int = Field(default=5, ge=1)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalysisConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)
```

(This preserves every field and `Field(...)` constraint already in the file — including `root_cause_granularity`, added in a prior in-progress change — and only adds `model_config` and `from_yaml`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3.12 -m pytest tests/test_config.py -v`
Expected: all tests PASS (existing + 5 new).

- [ ] **Step 6: Run the full suite to confirm `extra="forbid"` doesn't break anything else**

Run: `python3.12 -m pytest tests/ -v`
Expected: all tests PASS. (Every existing `AnalysisConfig(...)` call site across `tests/` and `nce_analysis/` was checked during design — none passes an unrecognized field name.)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml nce_analysis/config.py tests/test_config.py
git commit -m "$(cat <<'EOF'
feat: add AnalysisConfig.from_yaml and reject unknown config fields

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Input file loading (`nce_analysis/io.py`)

**Files:**
- Modify: `pyproject.toml`
- Create: `nce_analysis/io.py`
- Create: `tests/test_io.py`

**Interfaces:**
- Consumes: nothing from `nce_analysis` (pure pandas/json/pathlib).
- Produces: `InputFormatError(Exception)`, `load_input(path: str | Path) -> pd.DataFrame` returning a DataFrame in the exact shape `pipeline.run` already expects (one row per wafer, `Measurement_Points: list[dict]` column). Task 3 (`cli.py`) imports both.

- [ ] **Step 1: Add the `pyarrow` dependency**

Edit `pyproject.toml` — add `"pyarrow>=14.0",` as the last entry in `dependencies` (after the `"pyyaml>=6.0",` line added in Task 1):

```toml
dependencies = [
    "pandas>=2.0",
    "pydantic>=2.0",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "shap>=0.44",
    "numpy>=1.24",
    "pyyaml>=6.0",
    "pyarrow>=14.0",
]
```

Run:
```bash
python3.12 -m pip install --user --break-system-packages -e ".[dev]"
```
Expected: install succeeds, pulling in `pyarrow`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_io.py`:

```python
import json

import pandas as pd
import pytest

from nce_analysis.io import InputFormatError, load_input


def _wafer_row(wafer_id, **overrides):
    row = {
        "PartID": "PART1",
        "WaferID": wafer_id,
        "StageID": "LITHO_M1",
        "StepID": "S1",
        "ToolID": "LITHO_A",
        "ChuckID": "CHK_1",
        "Execute_Time": "2026-01-01",
        "Pre_StageID_1": "CMP",
        "Pre_StepID_1": "3580.01",
        "Pre_ToolID_1": "CMP_01",
        "Pre_ChamberID_1": "ChamberA",
        "Pre_Execute_Time_1": "2025-12-01",
    }
    row.update(overrides)
    return row


def test_load_input_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("not a real data file")

    with pytest.raises(InputFormatError):
        load_input(path)


def test_load_input_csv_wide_format(tmp_path):
    row = _wafer_row("W1")
    row["Measurement_Points"] = json.dumps(
        [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0}]
    )
    csv_path = tmp_path / "data.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    result = load_input(csv_path)

    assert len(result) == 1
    points = result.iloc[0]["Measurement_Points"]
    assert points == [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0}]


def test_load_input_csv_long_format_groups_points_by_wafer(tmp_path):
    rows = [
        {**_wafer_row("W1"), "X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0},
        {**_wafer_row("W1"), "X_Posi": 6.0, "Y_Posi": 0.0, "NCE_Value": 5.0},
        {**_wafer_row("W2"), "X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 20.0},
    ]
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = load_input(csv_path)

    assert len(result) == 2
    w1_points = result.loc[result["WaferID"] == "W1", "Measurement_Points"].iloc[0]
    assert len(w1_points) == 2
    assert {"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0} in w1_points


def test_load_input_csv_missing_wafer_id_raises(tmp_path):
    rows = [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0}]
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with pytest.raises(InputFormatError):
        load_input(csv_path)


def test_load_input_csv_unrecognized_shape_raises(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame([{"foo": 1, "bar": 2}]).to_csv(csv_path, index=False)

    with pytest.raises(InputFormatError):
        load_input(csv_path)


def test_load_input_parquet_round_trips_measurement_points_as_list(tmp_path):
    row = _wafer_row("W1")
    row["Measurement_Points"] = [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0}]
    parquet_path = tmp_path / "data.parquet"
    pd.DataFrame([row]).to_parquet(parquet_path)

    result = load_input(parquet_path)

    points = result.iloc[0]["Measurement_Points"]
    assert isinstance(points, list)
    assert points == [{"X_Posi": 0.0, "Y_Posi": 0.0, "NCE_Value": 16.0}]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3.12 -m pytest tests/test_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.io'`.

- [ ] **Step 4: Implement**

Create `nce_analysis/io.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3.12 -m pytest tests/test_io.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml nce_analysis/io.py tests/test_io.py
git commit -m "$(cat <<'EOF'
feat: add CSV/Parquet input loading with wide/long format detection

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: CLI entry point (`nce_analysis/cli.py` + `nce_analysis/__main__.py`)

**Files:**
- Create: `nce_analysis/cli.py`
- Create: `nce_analysis/__main__.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `nce_analysis.io.InputFormatError`, `nce_analysis.io.load_input(path) -> pd.DataFrame` (Task 2); `nce_analysis.config.AnalysisConfig`, `AnalysisConfig.from_yaml(path) -> AnalysisConfig` (Task 1); `nce_analysis.preprocessing.base.PreprocessingError` (existing); `nce_analysis.pipeline.run(raw_df, config) -> AnalysisResult` (existing, unchanged).
- Produces: `main(argv: list[str] | None = None) -> int`. `python -m nce_analysis` runs it via `__main__.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
import json

import pandas as pd

from nce_analysis.cli import main


def _write_minimal_csv(path):
    row = {
        "PartID": "P1",
        "WaferID": "W1",
        "StageID": "LITHO_M1",
        "StepID": "S1",
        "ToolID": "LITHO_A",
        "ChuckID": "CHK_1",
        "Execute_Time": "2026-01-01",
        "X_Posi": 0.0,
        "Y_Posi": 0.0,
        "NCE_Value": 5.0,
        "Pre_StageID_1": "CMP",
        "Pre_StepID_1": "3580.01",
        "Pre_ToolID_1": "CMP_01",
        "Pre_ChamberID_1": "ChamberA",
        "Pre_Execute_Time_1": "2025-12-01",
    }
    pd.DataFrame([row]).to_csv(path, index=False)


def test_main_writes_result_json_to_output_file(tmp_path):
    csv_path = tmp_path / "data.csv"
    out_path = tmp_path / "result.json"
    _write_minimal_csv(csv_path)

    exit_code = main(["--input", str(csv_path), "--output", str(out_path)])

    assert exit_code == 0
    result = json.loads(out_path.read_text())
    assert "Summary" in result
    assert "Details" in result


def test_main_prints_result_json_to_stdout_when_no_output(tmp_path, capsys):
    csv_path = tmp_path / "data.csv"
    _write_minimal_csv(csv_path)

    exit_code = main(["--input", str(csv_path)])

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert "Summary" in result


def test_main_applies_config_yaml_override(tmp_path):
    csv_path = tmp_path / "data.csv"
    out_path = tmp_path / "result.json"
    config_path = tmp_path / "config.yaml"
    _write_minimal_csv(csv_path)
    config_path.write_text("summary_top_n: 3\n")

    exit_code = main(
        ["--input", str(csv_path), "--config", str(config_path), "--output", str(out_path)]
    )

    assert exit_code == 0
    result = json.loads(out_path.read_text())
    assert result["Config_Used"]["summary_top_n"] == 3


def test_main_reports_error_for_unsupported_input_extension(tmp_path, capsys):
    bad_path = tmp_path / "data.txt"
    bad_path.write_text("not real data")

    exit_code = main(["--input", str(bad_path)])

    assert exit_code == 1
    assert "Error loading input file" in capsys.readouterr().err


def test_main_reports_error_for_invalid_config_field(tmp_path, capsys):
    csv_path = tmp_path / "data.csv"
    config_path = tmp_path / "config.yaml"
    _write_minimal_csv(csv_path)
    config_path.write_text("typo_field: 1\n")

    exit_code = main(["--input", str(csv_path), "--config", str(config_path)])

    assert exit_code == 1
    assert "Error loading config file" in capsys.readouterr().err


def test_main_reports_error_for_preprocessing_failure(tmp_path, capsys):
    csv_path = tmp_path / "data.csv"
    # No Pre_*_i history columns at all -> discover_history_levels finds
    # none -> PreprocessingError inside pipeline.run.
    pd.DataFrame(
        [
            {
                "PartID": "P1",
                "WaferID": "W1",
                "StageID": "LITHO_M1",
                "StepID": "S1",
                "ToolID": "LITHO_A",
                "ChuckID": "CHK_1",
                "Execute_Time": "2026-01-01",
                "X_Posi": 0.0,
                "Y_Posi": 0.0,
                "NCE_Value": 5.0,
            }
        ]
    ).to_csv(csv_path, index=False)

    exit_code = main(["--input", str(csv_path)])

    assert exit_code == 1
    assert "Error running analysis pipeline" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nce_analysis.cli'`.

- [ ] **Step 3: Implement**

Create `nce_analysis/cli.py`:

```python
import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from nce_analysis import pipeline
from nce_analysis.config import AnalysisConfig
from nce_analysis.io import InputFormatError, load_input
from nce_analysis.preprocessing.base import PreprocessingError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nce_analysis",
        description="Run the NCE cross-stage root-cause analysis pipeline on a wafer batch file.",
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Input wafer batch file (.csv or .parquet)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML config file (optional; AnalysisConfig defaults if omitted)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to write result JSON (optional; prints to stdout if omitted)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        raw_df = load_input(args.input)
    except (InputFormatError, FileNotFoundError) as e:
        print(f"Error loading input file: {e}", file=sys.stderr)
        return 1

    try:
        config = AnalysisConfig.from_yaml(args.config) if args.config else AnalysisConfig()
    except (FileNotFoundError, ValidationError) as e:
        print(f"Error loading config file: {e}", file=sys.stderr)
        return 1

    try:
        result = pipeline.run(raw_df, config)
    except PreprocessingError as e:
        print(f"Error running analysis pipeline: {e}", file=sys.stderr)
        return 1

    output_json = result.model_dump_json(indent=2)
    if args.output:
        args.output.write_text(output_json, encoding="utf-8")
    else:
        print(output_json)
    return 0
```

Create `nce_analysis/__main__.py`:

```python
from nce_analysis.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m pytest tests/test_cli.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Smoke-test the actual `python -m nce_analysis` entry point**

Run:
```bash
python3.12 -c "
import pandas as pd
pd.DataFrame([{
    'PartID': 'P1', 'WaferID': 'W1', 'StageID': 'LITHO_M1', 'StepID': 'S1',
    'ToolID': 'LITHO_A', 'ChuckID': 'CHK_1', 'Execute_Time': '2026-01-01',
    'X_Posi': 0.0, 'Y_Posi': 0.0, 'NCE_Value': 5.0,
    'Pre_StageID_1': 'CMP', 'Pre_StepID_1': '3580.01',
    'Pre_ToolID_1': 'CMP_01', 'Pre_ChamberID_1': 'ChamberA',
    'Pre_Execute_Time_1': '2025-12-01',
}]).to_csv('/tmp/smoke_test.csv', index=False)
"
python3.12 -m nce_analysis --input /tmp/smoke_test.csv
rm /tmp/smoke_test.csv
```
Expected: prints a JSON object with top-level keys `Summary`, `Details`, `Insufficient_Sample_Hotspots`, `Generated_At`, `Config_Used`, and exits with code 0 (check with `echo $?` if unsure).

- [ ] **Step 6: Commit**

```bash
git add nce_analysis/cli.py nce_analysis/__main__.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
feat: add python -m nce_analysis CLI entry point

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `examples/` CLI-demo fixtures

**Files:**
- Create: `examples/generate_sample_files.py`
- Create (generated by the script, then committed): `examples/sample_data.csv`, `examples/sample_data.parquet`, `examples/sample_config.yaml`

**Interfaces:**
- Consumes: `examples.sample_data.build_sample_dataframe() -> pd.DataFrame` (existing, unchanged).
- Produces: the three fixture files, consumed by Task 5's README edits and by any user following the Quick Start.

- [ ] **Step 1: Create the generator script**

Create `examples/generate_sample_files.py`:

```python
"""Regenerate the CLI-demo fixture files (sample_data.csv, sample_data.parquet,
sample_config.yaml) from sample_data.build_sample_dataframe().

Run this after changing examples/sample_data.py to refresh the fixtures;
the fixtures themselves are committed so the CLI quick-start works without
running this script first.

Usage:
    python3.12 examples/generate_sample_files.py
"""

from pathlib import Path

import pandas as pd
from sample_data import build_sample_dataframe

EXAMPLES_DIR = Path(__file__).parent


def main() -> None:
    raw_df = build_sample_dataframe()

    raw_df.to_parquet(EXAMPLES_DIR / "sample_data.parquet")

    long_df = raw_df.explode("Measurement_Points", ignore_index=True)
    points = pd.json_normalize(long_df["Measurement_Points"])
    long_df = pd.concat(
        [long_df.drop(columns=["Measurement_Points"]).reset_index(drop=True), points],
        axis=1,
    )
    long_df.to_csv(EXAMPLES_DIR / "sample_data.csv", index=False)

    (EXAMPLES_DIR / "sample_config.yaml").write_text(
        "# Overrides AnalysisConfig defaults; any field not listed here keeps\n"
        "# its default value (see nce_analysis/config.py).\n"
        "summary_top_n: 3\n"
    )

    print(f"Wrote {EXAMPLES_DIR / 'sample_data.csv'}")
    print(f"Wrote {EXAMPLES_DIR / 'sample_data.parquet'}")
    print(f"Wrote {EXAMPLES_DIR / 'sample_config.yaml'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to generate the fixtures**

Run:
```bash
cd /mnt/c/Users/y4lin/gemini_repo/nce_analysis_service
python3.12 examples/generate_sample_files.py
```
Expected: prints the three "Wrote ..." lines; `examples/sample_data.csv`, `examples/sample_data.parquet`, `examples/sample_config.yaml` now exist.

- [ ] **Step 3: Verify the fixtures actually load and run through the full CLI**

Run:
```bash
cd /mnt/c/Users/y4lin/gemini_repo/nce_analysis_service
python3.12 -m nce_analysis --input examples/sample_data.csv --output /tmp/csv_result.json
python3.12 -m nce_analysis --input examples/sample_data.parquet --output /tmp/parquet_result.json
python3.12 -m nce_analysis --input examples/sample_data.csv --config examples/sample_config.yaml --output /tmp/config_result.json
python3.12 -c "
import json
csv_r = json.load(open('/tmp/csv_result.json'))
pq_r = json.load(open('/tmp/parquet_result.json'))
cfg_r = json.load(open('/tmp/config_result.json'))
assert csv_r['Summary'], 'expected at least one Summary entry from sample_data.csv'
assert csv_r['Config_Used']['summary_top_n'] == 5
assert cfg_r['Config_Used']['summary_top_n'] == 3
assert any(d['Suspect_Pre_ToolID'] == 'CMP_01' for d in csv_r['Details'])
assert any(d['Suspect_Pre_ToolID'] == 'CMP_01' for d in pq_r['Details'])
print('OK: both fixtures load and the known CMP_01 root cause is found')
"
rm /tmp/csv_result.json /tmp/parquet_result.json /tmp/config_result.json
```
Expected: `OK: both fixtures load and the known CMP_01 root cause is found` — this confirms both the CSV long-format reconstruction path and the Parquet native path produce a batch where the pipeline still finds the same injected root cause (`CMP_01`/`ChamberA`) that `examples/run_example.py`'s docstring already describes, and that `--config` actually changes `summary_top_n`.

- [ ] **Step 4: Commit**

```bash
git add examples/generate_sample_files.py examples/sample_data.csv examples/sample_data.parquet examples/sample_config.yaml
git commit -m "$(cat <<'EOF'
feat: add committed CSV/Parquet/YAML fixtures for the CLI quick start

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update docs (`run_example.py` docstring, `README.md`, `README.zh-TW.md`)

**Files:**
- Modify: `examples/run_example.py`
- Modify: `README.md`
- Modify: `README.zh-TW.md`

**Interfaces:** None — documentation only, no code behavior changes.

- [ ] **Step 1: Point `run_example.py`'s docstring at the CLI path**

In `examples/run_example.py`, replace the module docstring (lines 1-5):

Old:
```python
"""Run the sample wafer batch through nce_analysis.pipeline end to end.

Usage:
    python3.12 examples/run_example.py
"""
```

New:
```python
"""Run the sample wafer batch through nce_analysis.pipeline end to end.

Usage:
    python3.12 examples/run_example.py

For the file-based / CLI path (CSV or Parquet input, YAML config) instead
of this in-process API call, see examples/sample_data.csv,
examples/sample_data.parquet, examples/sample_config.yaml, and the
"Quick start" section of README.md.
"""
```

- [ ] **Step 2: Update `README.md`'s Quick start section**

In `README.md`, after the existing Quick start code block and its explanatory paragraph/bullets (ending at "...doubles as a worked example of the input shape described below."), insert a new subsection before `## Using it on your own data`:

Old:
```markdown
Read `examples/sample_data.py` alongside the output — its module docstring
explains exactly which rows are supposed to produce which result, which
doubles as a worked example of the input shape described below.

## Using it on your own data
```

New:
```markdown
Read `examples/sample_data.py` alongside the output — its module docstring
explains exactly which rows are supposed to produce which result, which
doubles as a worked example of the input shape described below.

### CLI quick start

The same sample batch is also available as committed CSV/Parquet fixtures,
runnable straight through the command-line entry point instead of writing
Python:

```bash
python3.12 -m nce_analysis --input examples/sample_data.csv \
    --config examples/sample_config.yaml --output /tmp/result.json
cat /tmp/result.json
```

`examples/sample_data.parquet` works the same way (`--input
examples/sample_data.parquet`). Run `python3.12 examples/generate_sample_files.py`
to regenerate all three fixture files after changing `examples/sample_data.py`.

## Using it on your own data
```

- [ ] **Step 3: Add a CLI usage subsection to `README.md`**

In `README.md`, after the closing of the "Using it on your own data" Python code block (the ` ```  ` line right after `print(result.model_dump_json(indent=2))`), insert a new subsection before `### Input shape`:

Old:
```markdown
result = pipeline.run(raw_df, config)

print(result.model_dump_json(indent=2))
```

### Input shape
```

New:
```markdown
result = pipeline.run(raw_df, config)

print(result.model_dump_json(indent=2))
```

Or from the command line, without writing any Python:

```bash
python3.12 -m nce_analysis --input your_data.csv --config your_config.yaml --output result.json
```

- `--input`: required. A `.csv` or `.parquet` file. CSV accepts two shapes,
  auto-detected by column presence: **wide** (a `Measurement_Points` column
  holding a JSON-encoded list per row, matching the in-memory shape exactly)
  or **long** (one row per measurement point, with `X_Posi`/`Y_Posi`/`NCE_Value`
  columns directly and a `WaferID` column to group rows back into one wafer
  per row). Parquet only supports the wide shape (nested columns are native
  to Parquet, which is the point of using it).
- `--config`: optional. A YAML file with any subset of the `AnalysisConfig`
  fields listed below (e.g. `summary_top_n: 3`); omitted fields keep their
  default. Omit `--config` entirely to run with all defaults.
- `--output`: optional. Path to write the result JSON. Omit it to print the
  JSON to stdout instead.

### Input shape
```

- [ ] **Step 4: Update `README.zh-TW.md`'s Quick start section (mirrors Step 2)**

In `README.zh-TW.md`, after the existing Quick start bullets:

Old:
```markdown
建議對照著輸出結果閱讀 `examples/sample_data.py`——它的 module docstring
清楚說明了哪些資料列應該產生哪種結果，同時也是下方「輸入資料格式」的
實例示範。

## 在你自己的資料上使用
```

New:
```markdown
建議對照著輸出結果閱讀 `examples/sample_data.py`——它的 module docstring
清楚說明了哪些資料列應該產生哪種結果，同時也是下方「輸入資料格式」的
實例示範。

### CLI 快速開始

同一批範例資料也提供了 CSV/Parquet 版本（已 commit 進 repo），可以直接
透過命令列介面執行，不需要寫任何 Python：

```bash
python3.12 -m nce_analysis --input examples/sample_data.csv \
    --config examples/sample_config.yaml --output /tmp/result.json
cat /tmp/result.json
```

`examples/sample_data.parquet`用法相同（`--input
examples/sample_data.parquet`）。若修改了 `examples/sample_data.py`，可執行
`python3.12 examples/generate_sample_files.py` 重新產生這三個 fixture 檔案。

## 在你自己的資料上使用
```

- [ ] **Step 5: Add a CLI usage subsection to `README.zh-TW.md` (mirrors Step 3)**

In `README.zh-TW.md`, after the closing of the Python code block:

Old:
```markdown
result = pipeline.run(raw_df, config)

print(result.model_dump_json(indent=2))
```

### 輸入資料格式
```

New:
```markdown
result = pipeline.run(raw_df, config)

print(result.model_dump_json(indent=2))
```

或是不寫 Python，直接用命令列執行：

```bash
python3.12 -m nce_analysis --input your_data.csv --config your_config.yaml --output result.json
```

- `--input`：必填。`.csv` 或 `.parquet` 檔案。CSV 支援兩種格式，依欄位自動
  偵測：**wide**（有 `Measurement_Points` 欄位，內容是每列一筆 JSON 編碼的
  清單，跟記憶體中的形狀完全一致）或 **long**（一列一個量測點，直接有
  `X_Posi`/`Y_Posi`/`NCE_Value` 欄位，並用 `WaferID` 欄位把同一片晶圓的多筆
  量測點重新分組）。Parquet 只支援 wide 格式（巢狀欄位是 Parquet 原生支援
  的，這也是選用 Parquet 的意義所在）。
- `--config`：選填。YAML 檔案，可只列出 `AnalysisConfig` 部分欄位（例如
  `summary_top_n: 3`）；沒列出的欄位沿用預設值。完全不給 `--config` 就是
  全部用預設值執行。
- `--output`：選填。輸出結果 JSON 的檔案路徑；不給的話直接印到 stdout。

### 輸入資料格式
```

- [ ] **Step 6: Verify the doc edits render correctly and the CLI commands in them actually work**

Run:
```bash
cd /mnt/c/Users/y4lin/gemini_repo/nce_analysis_service
python3.12 -m nce_analysis --input examples/sample_data.csv \
    --config examples/sample_config.yaml --output /tmp/result.json
cat /tmp/result.json | python3.12 -m json.tool > /dev/null && echo "valid JSON"
rm /tmp/result.json
grep -n "CLI quick start\|python -m nce_analysis" README.md
grep -n "CLI 快速開始\|python -m nce_analysis" README.zh-TW.md
```
Expected: `valid JSON`, and both `grep` calls find the new sections.

- [ ] **Step 7: Commit**

```bash
git add examples/run_example.py README.md README.zh-TW.md
git commit -m "$(cat <<'EOF'
docs: document the CLI, YAML config, and CSV/Parquet input shapes

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Full-suite verification

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Run the full test suite**

Run: `python3.12 -m pytest tests/ -v`
Expected: every test passes, including all pre-existing tests plus the new `test_config.py`, `test_io.py`, and `test_cli.py` additions from Tasks 1-3.

- [ ] **Step 2: Confirm `git status` shows a clean, fully-committed feature**

Run: `git status`
Expected: no uncommitted changes belonging to this feature (`nce_analysis/io.py`, `nce_analysis/cli.py`, `nce_analysis/__main__.py`, `tests/test_io.py`, `tests/test_cli.py`, the three `examples/` fixtures, `generate_sample_files.py`, and the `pyproject.toml`/`config.py`/README/`run_example.py` edits should all show as committed). Any other pre-existing unrelated modified files in the working tree are out of scope for this feature and should be left as-is.
