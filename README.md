# nce-analysis

English | [繁體中文](README.zh-TW.md)

A pure-Python library that analyzes LITHO-stage NCE (Non-Correctable Error)
surface-flatness measurements, detects spatial anomaly hotspots on the wafer
surface, and traces each hotspot back to the upstream tool/chamber (CMP, CVD,
PVD, ...) most likely responsible — while filtering out anomalies that are
actually caused by the LITHO tool's own Tool/Chuck rather than an upstream
process.

There is no API/service layer: you call one function with a pandas
`DataFrame` and get back a typed result object.

## Install

```bash
python3.12 -m pip install --user --break-system-packages -e ".[dev]"
```

(No virtualenv is committed to this repo; `python3.12` is required — a bare
`python3` may resolve to a different interpreter on your machine.)

## Quick start

`examples/` contains a runnable, self-contained sample so you can see the
whole pipeline work before wiring up your own data:

```bash
python3.12 examples/run_example.py
```

This builds a synthetic batch of 33 wafers (`examples/sample_data.py`) and
runs it through `nce_analysis.pipeline.run`, printing:

- a confirmed root cause (`CMP_01`/`ChamberA`, classified `CHAMBER_DRIFT`)
  for a hotspot where NCE gets steadily worse over time,
- a hotspot correctly attributed to the LITHO tool's own chuck
  (`LITHO_CHUCK_ISSUE`) rather than to any upstream tool,
- a third hotspot reported as `Insufficient_Sample_Hotspots` because it has
  too few wafers to analyze with confidence.

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

```python
import pandas as pd
from nce_analysis import pipeline
from nce_analysis.config import AnalysisConfig

raw_df = pd.DataFrame([...])  # see "Input shape" below
config = AnalysisConfig()      # defaults, or override specific fields

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

One row per wafer. History columns are expanded positionally as
`Pre_<Field>_<N>`, where `_1` is the stage immediately before the LITHO step
being analyzed, `_2` the one before that, and so on. The number of history
levels `N` is discovered dynamically from the columns present — it does not
need to be the same for every wafer (a wafer with fewer prior stages than the
batch's max `N` just has `None`/absent values for the higher levels).

| Column | Type | Meaning |
|---|---|---|
| `PartID` | str | Product/part identifier |
| `WaferID` | str | Wafer identifier |
| `StageID`, `StepID` | str | The LITHO stage/step being measured |
| `ToolID`, `ChuckID` | str | The LITHO tool and chuck that ran this wafer |
| `Execute_Time` | str/date | When the LITHO measurement was taken |
| `Measurement_Points` | list of `{X_Posi, Y_Posi, NCE_Value}` | Surface-flatness readings for this wafer |
| `Pre_StageID_i`, `Pre_StepID_i` | str | Upstream stage/step, `i`-th level back |
| `Pre_ToolID_i`, `Pre_ChamberID_i` | str | Upstream tool/chamber, `i`-th level back |
| `Pre_Execute_Time_i` | str/date | When the upstream step ran, `i`-th level back |

One wafer contributes one row per surface coordinate it was measured at
(i.e. `Measurement_Points` for a real wafer normally has many entries, one
per (X, Y) location on the reticle grid); `examples/sample_data.py` uses a
single point per wafer for readability.

### Key config knobs (`AnalysisConfig`)

| Field | Default | Effect |
|---|---|---|
| `spec_threshold` | `15.0` | `NCE_Value` above this counts as anomalous |
| `min_wafer_count` | `5` | Coordinates with fewer wafers than this are reported in `Insufficient_Sample_Hotspots` instead of analyzed |
| `hotspot_ratio_threshold` | `0.05` | Minimum anomalous-wafer ratio at a coordinate to count as a hotspot |
| `noise_filter_majority_threshold` | `0.5` | Share of anomalous wafers one LITHO chuck/tool must account for to be blamed on LITHO itself instead of upstream |
| `root_cause_strategy` | `"both"` | `"statistical"` (chi-square/Fisher), `"ml"` (SHAP on a decision tree), or `"both"` (cross-validates; disagreement sets `Requires_Manual_Review`) |
| `root_cause_granularity` | `"chamber"` | `"chamber"` keys suspects by `Pre_ToolID + Pre_ChamberID`; `"tool"` keys by `Pre_ToolID` alone |
| `drift_strategy` | `"regression_cusum"` | `"regression_cusum"` (linear trend + change-point detection) or `"correlation"` (simple Pearson correlation) |
| `alpha` | `0.05` | Significance threshold used by the statistical root-cause strategy |
| `summary_top_n` | `5` | How many suspects `AnalysisResult.Summary` keeps, sorted by confidence |

### Output (`AnalysisResult`)

- `Summary`: top suspects (deduped by tool/chamber/step/root-cause-type,
  sorted by `Confidence_Score` descending, truncated to `summary_top_n`) —
  the list to show a human first.
- `Details`: every suspect found, one entry per (coordinate × suspect ×
  root-cause type), before the summary dedup/truncation.
- `Insufficient_Sample_Hotspots`: coordinates with too few wafers to analyze;
  raw counts only, no confidence score.
- `Root_Cause_Type` is one of: `LITHO_TOOL_ISSUE`, `LITHO_CHUCK_ISSUE`,
  `LITHO_CHUCK_CONTAMINATION` (all three from the noise filter, meaning
  LITHO's own equipment is at fault), `SPECIFIC_CHAMBER_DEFECT` (upstream
  suspect, no time trend), `CHAMBER_DRIFT`, `CHAMBER_SUDDEN_SHIFT` (upstream
  suspect with a detected trend/change-point).

## Development

See `CLAUDE.md` for the test commands, TDD workflow, and internal pipeline
architecture. Full design rationale (why grouping is dynamic, noise-filter
thresholds, statistical fallback rules) lives in
`docs/superpowers/specs/2026-07-04-cross-stage-impact-analysis-design.md`.

Generic, domain-free statistical building blocks (association ranking,
uplift ranking, trend/change-point detection, dominance, ratio hotspots)
live in the `analysis_core` package and are reusable outside the NCE
domain; the `nce_analysis` strategies are thin adapters over them.
