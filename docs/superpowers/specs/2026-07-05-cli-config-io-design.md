# File-Based Input, YAML Config, and CLI — Design

Date: 2026-07-05
Status: Approved (pending user sign-off on written spec)

## 1. Purpose

Today `nce_analysis` only works as an in-process library call:
`pipeline.run(raw_df, config)`, where `raw_df` is a `pandas.DataFrame` the
caller must already have built in memory, and `config` is an `AnalysisConfig`
instance the caller must construct in Python. There is no way to point the
library at a data file on disk, no way to externalize a config as a
shareable file, and no command-line entry point.

This spec adds three things, all additive (no change to `pipeline.run`'s
signature or behavior):

1. **File-based input** — load a wafer batch from a `.csv` or `.parquet`
   file into the same `raw_df` shape `pipeline.run` already expects.
2. **YAML config** — load an `AnalysisConfig` from a YAML file, falling back
   to `AnalysisConfig()` defaults for any field the file omits.
3. **CLI** — `python -m nce_analysis --input <file> [--config <file>]
   [--output <file>]` runs the full pipeline and writes (or prints) the
   result JSON.

`examples/` is updated to demonstrate the new file-based path alongside the
existing programmatic one.

## 2. Scope

**In scope**: reading input from CSV/Parquet, reading config from YAML,
one CLI entry point, updated examples/README.

**Out of scope**: a REST/service layer (explicitly excluded by the
project's overall design — see `2026-07-04-cross-stage-impact-analysis-design.md`
§1), CLI flags to override individual config fields (config is YAML-only),
console-script installation (`python -m nce_analysis` only), writing output
in any format other than JSON, reading input from any source other than a
local CSV/Parquet file.

## 3. Input File Loading (`nce_analysis/io.py`)

### The nested-column problem

The existing input contract (`docs/.../2026-07-04-...-design.md` §2) has one
row per wafer with `Measurement_Points: list[{X_Posi, Y_Posi, NCE_Value}]` —
a nested column. Parquet supports nested `list<struct>` columns natively, so
a Parquet file can carry the exact same shape `pipeline.run` expects
in-memory ("wide format"). CSV is flat text and cannot store a list/struct
directly, so CSV supports **two** formats, auto-detected by column presence:

- **Wide**: a `Measurement_Points` column present, holding a JSON-encoded
  string per row (e.g. `'[{"X_Posi":0.0,"Y_Posi":0.0,"NCE_Value":16.0}]'`).
  Parsed with `json.loads` per cell.
- **Long**: no `Measurement_Points` column, but `X_Posi`, `Y_Posi`, and
  `NCE_Value` columns present directly — one row per measurement point
  instead of one row per wafer. Requires a `WaferID` column to group rows
  back into one `Measurement_Points` list per wafer. This is the more
  natural, spreadsheet-friendly CSV shape.

If a CSV has a `Measurement_Points` column, wide format takes priority
(checked first) even if `X_Posi`/`Y_Posi`/`NCE_Value` columns also happen to
be present. If neither shape matches, loading fails with `InputFormatError`
naming both accepted shapes.

Long-format reconstruction takes wafer-level columns (`PartID`, `Pre_*_i`
history columns, etc. — anything that isn't `X_Posi`/`Y_Posi`/`NCE_Value`)
from the **first** row of each `WaferID` group; it does not cross-validate
that those columns are identical across every row in the group. This mirrors
the input contract's own assumption (one wafer has one process history, but
may contribute multiple measurement points).

Parquet only supports the wide/native shape — long-format Parquet is not
supported, since avoiding the CSV nested-column workaround is the point of
offering Parquet at all.

### The `numpy.ndarray` gotcha (verified empirically)

`pd.read_parquet` round-trips a `list<struct>` column back as a
`numpy.ndarray` per cell, **not** a Python `list` (individual elements are
still plain `dict`). `wide_history_reshape.explode_measurement_points`
checks `isinstance(points, list)` — an `ndarray` fails that check, so every
wafer would silently be treated as having an empty `Measurement_Points` and
the whole batch would raise `PreprocessingError` ("nothing to analyze").
`load_input` must therefore normalize every cell to a real `list` right
after `pd.read_parquet` (`df["Measurement_Points"].map(list)`), before the
DataFrame ever reaches `preprocessing/`.

### API

```python
class InputFormatError(Exception):
    """Raised when an input file's on-disk shape can't be mapped to the
    Measurement_Points-per-wafer contract pipeline.run expects."""


def load_input(path: str | Path) -> pd.DataFrame:
    """Dispatch on file extension (.csv -> CSV wide/long detection,
    .parquet/.pq -> native nested read + ndarray->list normalization).
    Returns a DataFrame in the same shape pipeline.run already accepts."""
```

Internal helpers (not part of the public surface): `_normalize_csv`,
`_reconstruct_from_long`.

## 4. YAML Config Loading (`AnalysisConfig.from_yaml`)

Added directly to `nce_analysis/config.py` as a classmethod, rather than a
separate loader module — "how do I load this config" belongs next to the
class it loads, the same way `AnalysisResult.model_dump_json()` lives on the
model it serializes.

```python
class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ... existing fields, unchanged ...

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalysisConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)
```

- YAML content is a flat key-value mapping onto `AnalysisConfig`'s existing
  fields (e.g. `spec_threshold: 20.0`). Fields the file omits keep their
  `AnalysisConfig` default.
- An empty YAML file (`yaml.safe_load` returns `None`) is equivalent to
  supplying no overrides at all.
- **Behavior change, applied model-wide, not just to YAML loading**:
  `AnalysisConfig` currently has no `extra` setting, so pydantic v2's
  default (`extra="ignore"`) silently drops unknown keyword arguments
  (verified: `AnalysisConfig(typo_field=99)` succeeds today and just
  discards `typo_field`). Setting `extra="forbid"` makes *any* construction
  path — YAML or direct Python — raise `pydantic.ValidationError` on an
  unrecognized field name. This is deliberate: it's the only way to catch a
  misspelled key in a YAML file, and no existing test or call site passes an
  extra/misspelled field today, so nothing currently passing changes
  behavior.
- File not found, and field-level validation errors (wrong type, out of the
  `Field(ge=..., le=...)` bounds, misspelled key) are not wrapped — they
  propagate as `FileNotFoundError` / `pydantic.ValidationError` respectively,
  which already carry clear, specific messages.

## 5. CLI (`nce_analysis/cli.py` + `nce_analysis/__main__.py`)

Single supported invocation form: `python -m nce_analysis` (via
`__main__.py`; no `[project.scripts]` console-script entry, and `cli.py`
itself has no `__main__` guard — one canonical entry point, not two).

```
python -m nce_analysis --input FILE [--config FILE] [--output FILE]

--input   required. Path to a .csv or .parquet wafer batch file.
--config  optional. Path to a YAML AnalysisConfig file. Omitted -> AnalysisConfig() defaults.
--output  optional. Path to write the result JSON. Omitted -> printed to stdout.
```

```python
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

Only the three known, documented failure boundaries (bad input file shape,
bad config file, `PreprocessingError` from the pipeline itself — e.g. an
all-empty batch) are caught and turned into a clean stderr message + exit
code 1. Any other exception is a bug, not a user-input problem, and is
allowed to propagate as a normal Python traceback rather than being masked
by a generic "something went wrong" message.

## 6. `examples/` Adjustments

```
examples/
├── sample_data.py            # unchanged: build_sample_dataframe()
├── run_example.py            # unchanged logic; docstring gains a pointer to the CLI path
├── generate_sample_files.py  # new: regenerates the three fixtures below from sample_data.py
├── sample_data.csv           # new, committed fixture: long format
├── sample_data.parquet       # new, committed fixture: wide/native format
└── sample_config.yaml        # new, committed fixture: overrides summary_top_n: 3
```

- `generate_sample_files.py` explodes `build_sample_dataframe()`'s output
  into one-row-per-point for `sample_data.csv` (exercises the CSV long-format
  reconstruction path) and writes it unmodified via `to_parquet` for
  `sample_data.parquet` (exercises the wide/native path and the
  `numpy.ndarray` normalization). Between the two committed fixtures, both
  `load_input` code paths get exercised by simply following the README.
- `sample_config.yaml` overrides only `summary_top_n: 3`, so running the CLI
  with vs. without `--config` visibly changes the output (proves the file is
  actually read) without changing which root causes are found (keeps the
  existing README narrative about the sample batch's three scenarios valid).
- Fixtures are committed rather than generated on first use, so `python -m
  nce_analysis --input examples/sample_data.csv ...` works immediately;
  `generate_sample_files.py` is the tool for regenerating them after
  `sample_data.py` changes.
- `README.md` / `README.zh-TW.md`: Quick start gains a CLI invocation
  example; "Using it on your own data" gains a subsection on the CSV
  wide/long shapes, Parquet, and YAML config field names.

## 7. Error Handling Summary

Consistent with the existing project rule (validate only at boundaries,
trust internal data once past them):

- `nce_analysis/io.py`: `InputFormatError` for an unrecognized file
  extension, a CSV matching neither the wide nor long shape, or a
  long-format CSV missing `WaferID`.
- `nce_analysis/config.py`: no new exception type — reuses
  `pydantic.ValidationError` (now including unknown-field errors, via
  `extra="forbid"`) and the stdlib's own `FileNotFoundError`.
- `nce_analysis/pipeline.py`: unchanged — still raises `PreprocessingError`
  for the cases already documented in the existing design (all-wafers-empty
  batch, zero discovered history columns).
- `nce_analysis/cli.py` is the outermost boundary: catches exactly the three
  exception types above (per stage) and converts them to `stderr` + exit
  code 1; everything else propagates.

## 8. Dependencies (`pyproject.toml`)

Add to `[project.dependencies]`: `pyyaml` (YAML parsing), `pyarrow` (parquet
read/write engine for pandas).

## 9. Testing Plan

- `tests/test_config.py`: `from_yaml` with a partial-override file, a
  missing file (`FileNotFoundError`), a file with an unknown field
  (`ValidationError`), and an empty file (all defaults).
- `tests/test_io.py` (new): CSV wide, CSV long, Parquet (including an
  explicit assertion that `Measurement_Points` cells are `list`, not
  `numpy.ndarray`, after loading — regression test for §3's gotcha),
  unsupported extension, CSV matching neither shape, long-format CSV
  missing `WaferID`.
- `tests/test_cli.py` (new): end-to-end `main()` call against a `tmp_path`
  CSV/Parquet fixture — exit code 0 and valid `AnalysisResult` JSON on
  success (to a file and to stdout), `--config` visibly changing
  `summary_top_n` in the output, and exit code 1 + stderr message for each
  of the three error boundaries in §7.
