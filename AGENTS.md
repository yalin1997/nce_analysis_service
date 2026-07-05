# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pure-Python library (no API/service layer) that analyzes LITHO-stage NCE
(Non-Correctable Error) surface-flatness measurements, detects spatial anomaly
hotspots, and traces them back to the upstream tool/chamber (CMP, CVD, PVD,
...) most likely responsible — while filtering out anomalies actually caused
by the LITHO tool's own Tool/Chuck. Input scale is hundreds to low-thousands
of wafers per call; there's no need to optimize for larger data.

Full design rationale lives in `docs/superpowers/specs/2026-07-04-cross-stage-impact-analysis-design.md`
(spec) and `docs/superpowers/plans/2026-07-04-cross-stage-impact-analysis.md`
(task-by-task TDD implementation plan). Read the spec before changing
pipeline-stage behavior — several rules below (grouping key, noise-filter
thresholds, chi-square/Fisher fallback) come from decisions recorded there.

## Commands

There is no committed virtualenv, and a bare `python3` may resolve to an
unrelated project's interpreter. Use `python3.12` explicitly:

```bash
# install (first time / after dependency changes)
python3.12 -m pip install --user --break-system-packages -e ".[dev]"

# full suite
python3.12 -m pytest tests/ -v

# single file / single test
python3.12 -m pytest tests/test_statistical.py -v
python3.12 -m pytest tests/test_pipeline_e2e.py::test_pipeline_identifies_injected_root_cause -v
```

When changing behavior in this repo, follow a red/green TDD loop: update or
add the test first, confirm it fails, then change production code until the
full suite is green — don't report a fix as done without an actual passing
`pytest` run.

## Architecture

Pipeline of swappable strategy components, each stage an ABC with one
concrete default, selected per-call via a single `AnalysisConfig`:

```
nce_analysis/
├── preprocessing/wide_history_reshape.py   # WideHistoryReshape (default)
├── hotspot/ratio_threshold.py              # RatioThreshold (default)
├── noise_filter/majority_rule.py           # MajorityRule (default)
├── root_cause/{statistical,ml,both}.py     # StatisticalStrategy / MLStrategy / BothStrategy
├── drift/{regression_cusum,correlation}.py # RegressionCusum (default) / Correlation
├── config.py         # AnalysisConfig — picks the concrete strategy per stage
├── pipeline.py        # run(raw_df, config) -> AnalysisResult — orchestrates all stages
├── schema.py           # pydantic input/output models (RootCauseDetail, AnalysisResult, ...)
└── result.py            # aggregate_results: Details -> deduped/sorted Summary
```

Data flow (`pipeline.run`): `Preprocessing → HotspotDetection → NoiseFilter →
[per surviving hotspot coordinate] → RootCauseAnalysis (per (Pre_StageID,
Pre_StepID) group) → DriftAnalysis → ResultAggregation`.

Entry point: `result: AnalysisResult = nce_analysis.pipeline.run(raw_df,
config)`; `result.model_dump_json()` for JSON output.

### The input shape and why grouping is dynamic

Input is already joined upstream: one row per wafer with history columns
expanded positionally (`Pre_StageID_1`, `Pre_StageID_2`, ... — `_1` is the
immediately preceding stage). The number of history levels `N` is **not
fixed** across datasets, and the positional suffix does **not** guarantee a
stage type — one wafer's `_1` might be CMP, another's might be CVD.

Because of this, `WideHistoryReshape` discovers history levels dynamically
via regex over the input columns (`discover_history_levels`), then unpivots
each level into long format rather than assuming a fixed schema. All
downstream grouping (in `pipeline.py`) groups by the **actual**
`(Pre_StageID, Pre_StepID)` values found in the data, never by the positional
index `_i`. This matters because a wafer can pass through the same stage type
twice (e.g. two CMP passes); `StepID`'s decimal suffix (`3580.01` vs
`3580.02`) keeps those passes in separate analysis groups so neither pass's
tool/chamber record is silently dropped or conflated with the other.

### Noise filter vs. root cause

`MajorityRule` runs on hotspot coordinates *before* root-cause analysis, to
rule out anomalies caused by LITHO's own equipment rather than an upstream
process: if one Chuck accounts for > `noise_filter_majority_threshold` share
of anomalous wafers at a coordinate, it's classified `LITHO_CHUCK_ISSUE` (or
`LITHO_CHUCK_CONTAMINATION` if that chuck's anomalies span multiple StageIDs);
else if one Tool dominates, `LITHO_TOOL_ISSUE`. These are terminal
classifications — they never reach root-cause/drift analysis. Only hotspots
that survive the noise filter proceed to `root_cause/`.

### Root cause granularity

`AnalysisConfig.root_cause_granularity` (`"tool"` or `"chamber"`, default
`"chamber"`) controls whether suspects are keyed by `Pre_ToolID` alone or
`Pre_ToolID + Pre_ChamberID` combined. All three places that need this
decision funnel through `root_cause/base.py` rather than re-deriving it:
`build_suspect_key`/`split_suspect_key` (used by `StatisticalStrategy` and
`MLStrategy` to build/decode the suspect key) and `matches_suspect` (used by
`pipeline.py` to pull a resolved candidate's full history for drift
analysis, via the same `build_suspect_key` encoding under the hood). Add any
new root-cause strategy or any new call site that needs "does this row
belong to suspect X" on top of these three functions instead of writing a
new `if granularity == "tool"` branch.

`StatisticalStrategy`: global chi-square test of independence, gated on
expected cell counts (`scipy.stats.contingency.expected_freq`); if any
expected count is < 5 the chi-square approximation is skipped in favor of
per-combination one-vs-rest Fisher exact tests directly (surfaced as the
`fisher_fallback` metric — a designed fallback, not an error path). The
winning suspect must have p < alpha *and* odds ratio > 1 (elevated risk, not
reduced).

`MLStrategy`: one-hot encode the suspect key, fit a shallow
`DecisionTreeClassifier`, use SHAP (`shap.TreeExplainer`) to find the suspect
group with the highest mean positive contribution toward `is_anomaly = 1`.

`BothStrategy`: runs both; if they agree on the same suspect, keeps the
higher confidence score; if they disagree (or only one finds a suspect),
returns both/the one candidate with `Requires_Manual_Review = True`.

### Drift analysis

Runs per surviving root-cause suspect, only when that suspect's combo has
enough history (< 3 rows falls back to `SPECIFIC_CHAMBER_DEFECT` with an
`insufficient_drift_sample` metric instead of calling the strategy).
`RegressionCusum` (default) does linear regression of `NCE_Value` on elapsed
`Pre_Execute_Time` for slope/significance (`CHAMBER_DRIFT`), plus a CUSUM
control-chart pass for abrupt change points (`CHAMBER_SUDDEN_SHIFT`);
otherwise `SPECIFIC_CHAMBER_DEFECT` (static, no time trend). Both drift
strategies guard against degenerate series (constant timestamps or constant
values) by returning early with an `insufficient_*_variation` metric rather
than dividing by zero / producing NaN.

### Error handling boundaries

Validation happens only at system boundaries; internal data is trusted once
past them (see spec §11):
- A wafer with empty `Measurement_Points` is skipped with a logged warning;
  an all-empty batch raises `PreprocessingError`.
- Zero discovered `Pre_*_i` history columns raises `PreprocessingError`
  explicitly (schema mismatch, not "no history").
- A hotspot coordinate below `min_wafer_count` is flagged
  `insufficient_sample`, reported in `AnalysisResult.Insufficient_Sample_Hotspots`
  with raw counts only — no confidence score is ever computed for it.
- Duplicate `(WaferID, Pre_StageID, Pre_StepID)` records (rework) keep the
  most recent `Pre_Execute_Time` row and log a warning rather than aborting.

### Output shape

`AnalysisResult.Details` is one `RootCauseDetail` per (coordinate ×
suspect-key × Root_Cause_Type); `Summary` (in `result.py`) merges details that
share the same `(Suspect_Pre_ToolID, Suspect_Pre_ChamberID,
Suspect_Pre_StepID, Root_Cause_Type)` key, unions their affected coordinates,
sorts by `Confidence_Score` desc, and truncates to `summary_top_n`.
`Root_Cause_Type` taxonomy: `LITHO_TOOL_ISSUE`, `LITHO_CHUCK_ISSUE`,
`LITHO_CHUCK_CONTAMINATION` (noise-filter only, never from root-cause/drift),
`SPECIFIC_CHAMBER_DEFECT`, `CHAMBER_DRIFT`, `CHAMBER_SUDDEN_SHIFT`.

`PRODUCT_TOOL_MISMATCH` (PartID × Tool/Chamber interaction) is explicitly out
of scope for v1 — don't add a PartID analysis dimension without discussing it
first.
