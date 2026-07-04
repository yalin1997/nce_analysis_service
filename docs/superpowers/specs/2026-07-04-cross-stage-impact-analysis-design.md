# Cross-Stage Impact Analysis Service — Design

Date: 2026-07-04
Status: Approved (pending user sign-off on written spec)

## 1. Purpose

Analyze LITHO-stage NCE (Non-Correctable Error) surface-flatness measurements, detect spatial
anomaly hotspots, and trace them back to the upstream tool/chamber (CMP, CVD, PVD, ...) most
likely responsible — while filtering out anomalies that are actually caused by the LITHO tool's
own Tool/Chuck rather than an upstream process.

Delivered as a **pure Python analysis library** (no API/service layer). Input scale: hundreds to
low-thousands of wafers per analysis call.

## 2. Input Data Shape

The service receives data that is **already joined** upstream — one row per wafer, with history
columns expanded positionally (`_1` = immediately preceding stage, `_2` = the one before that,
etc.). The number of history levels `N` is **not fixed** across datasets, and the positional
suffix does **not** guarantee a stage type (a wafer's `_1` might be CMP; another wafer's `_1`
might be CVD, depending on its actual process flow).

```
PartID, WaferID, StageID, StepID, ToolID, ChuckID, Execute_Time,
Measurement_Points: [{X_Posi, Y_Posi, NCE_Value}, ...],
Pre_StageID_1, Pre_StepID_1, Pre_ToolID_1, Pre_ChamberID_1, Pre_Execute_Time_1,
Pre_StageID_2, Pre_StepID_2, Pre_ToolID_2, Pre_ChamberID_2, Pre_Execute_Time_2,
... Pre_*_N
```

`X_Posi`/`Y_Posi` are already on a fixed 6mm grid and directly comparable across wafers — no
spatial tolerance/binning needed.

## 3. Architecture

Pipeline of swappable strategy components (each stage implements a common interface; a single
`AnalysisConfig` selects which concrete strategy is used per stage). Per-`Pre_StageID`-type
grouping (CMP / CVD / PVD analyzed independently) is done by an outer loop over the actual
`Pre_StageID` values found after preprocessing — not by positional index.

```
nce_analysis/
├── preprocessing/
│   ├── base.py             # PreprocessingStrategy interface
│   └── wide_history_reshape.py   # default: explode points + unpivot Pre_*_i groups
├── hotspot/
│   ├── base.py             # HotspotStrategy interface
│   └── ratio_threshold.py  # default: spec threshold + anomaly-ratio aggregation
├── noise_filter/
│   ├── base.py             # NoiseFilterStrategy interface
│   └── majority_rule.py    # default: >50% Tool/Chuck rule + cross-stage contamination check
├── root_cause/
│   ├── base.py             # RootCauseStrategy interface
│   ├── statistical.py      # Chi-square / Fisher exact test
│   └── ml.py                # Decision Tree / XGBoost + SHAP
├── drift/
│   ├── base.py             # DriftStrategy interface
│   ├── regression_cusum.py
│   └── correlation.py
├── config.py                # AnalysisConfig
├── pipeline.py               # Orchestrator, assembles strategies from config
├── schema.py                 # Pydantic input/output models
└── result.py                 # ResultAggregator

tests/
└── ... (per-component unit tests + one end-to-end fixture test)
```

Data flow: `Preprocessing → HotspotDetection → NoiseFilter → RootCauseAnalysis (per Pre_StageID
group) → DriftAnalysis → ResultAggregation`.

## 4. Preprocessing

Standard output columns (long format): `PartID, WaferID, StageID, StepID, ToolID, ChuckID,
Execute_Time, X_Posi, Y_Posi, NCE_Value, Pre_StageID, Pre_StepID, Pre_ToolID, Pre_ChamberID,
Pre_Execute_Time`.

Default `WideHistoryReshape` strategy:

1. **Discover history levels dynamically** via regex `^Pre_(StageID|StepID|ToolID|ChamberID|Execute_Time)_(\d+)$`
   over the input columns — do not assume a fixed N.
2. **Explode** `Measurement_Points` into one row per `(WaferID, X_Posi, Y_Posi, NCE_Value)`,
   carrying all wafer-level columns (including every `Pre_*_i` group) along.
3. **Unpivot per history level**: for each discovered index `i`, slice out that group's `Pre_*_i`
   columns, rename to the canonical (non-suffixed) names, and pair with the exploded measurement
   rows. Rows where `Pre_StageID_i` is null (wafer had fewer prior stages than this batch's max N)
   are dropped.
4. **Concatenate** all per-level slices into the final long table. The same `(WaferID, X_Posi,
   Y_Posi)` point will appear once per upstream stage it has a recorded history for (e.g. once for
   CMP, once for CVD).

Downstream grouping by upstream stage type uses the **actual `Pre_StageID` value**, never the
positional suffix.

## 5. Hotspot Detection (`ratio_threshold`)

1. De-duplicate the long table down to one row per `(WaferID, X_Posi, Y_Posi)` (undo the
   per-history-level expansion) before computing hotspot statistics, to avoid double counting.
2. `is_anomaly = NCE_Value > spec_threshold` (config, default 15).
3. Group by `(X_Posi, Y_Posi)`: `anomaly_ratio = anomalous_wafer_count / total_wafer_count`.
   Coordinates with `total_wafer_count < min_wafer_count` (config, default 5) are excluded from
   ranking (insufficient sample).
4. Rank by `anomaly_ratio` desc (tiebreak: `anomalous_wafer_count` desc). Coordinates at or above
   `hotspot_ratio_threshold` (config, default 0.05) are treated as hotspots.

## 6. Noise Filter (`majority_rule`)

Runs only on hotspot coordinates, to rule out anomalies that are actually a LITHO Tool/Chuck
problem rather than an upstream issue.

```
for each hotspot coordinate:
    anomalous_wafers = anomalous wafers at this coordinate
    chuck_share = share of anomalous_wafers from the single most common ChuckID
    if chuck_share > noise_filter_majority_threshold (default 0.5):
        if that Chuck's anomalies span > 1 distinct StageID:
            classify LITHO_CHUCK_CONTAMINATION (chuck needs cleaning)
        else:
            classify LITHO_CHUCK_ISSUE
        exclude from root-cause candidates
        continue
    tool_share = share of anomalous_wafers from the single most common ToolID
    if tool_share > noise_filter_majority_threshold:
        classify LITHO_TOOL_ISSUE
        exclude from root-cause candidates
        continue
    # otherwise: passes filter, becomes a root-cause candidate
```

## 7. Root Cause Analysis (per surviving hotspot, per `Pre_StageID` group)

### `statistical` strategy
1. Build a contingency table: `Pre_ToolID + Pre_ChamberID` combination × anomalous/normal.
2. Global Chi-square test of independence; if any expected cell count < 5, fall back to Fisher's
   exact test automatically (not an error path — a built-in behavior).
3. If globally significant (`p < alpha`, default 0.05): run one-vs-rest Fisher exact tests per
   combination; the suspect is the combination with the smallest p-value **and** odds ratio > 1
   (elevated, not reduced, anomaly rate).
4. `Confidence_Score = (1 - p_value) * 100`. `Metrics` includes p-value, odds ratio, sample counts.

### `ml` strategy
1. Features: `Pre_ToolID`, `Pre_ChamberID` (categorical). Label: `is_anomaly`.
2. Train a shallow Decision Tree (interpretability) or XGBoost (when interaction effects matter),
   compute SHAP values.
3. Identify the specific category value with the highest positive SHAP contribution toward
   `is_anomaly = 1` (not just "ToolID is important" in the abstract).
4. `Confidence_Score` = normalized contribution share of the top category, scaled to percentage.

### `both` (cross-validation)
Run both strategies independently. If they agree on the same suspect combination, average/boost
the confidence score and mark as cross-validated. If they disagree, report both candidates and set
`Requires_Manual_Review = true`.

## 8. Drift Analysis (on root-cause suspects)

### `regression_cusum` strategy
1. Linear regression of `NCE_Value` on `Pre_Execute_Time` (elapsed) for wafers processed by the
   suspect Tool+Chamber — a significant positive slope indicates gradual aging/drift.
2. CUSUM control-chart analysis on the same series to detect sudden change points (e.g. a PM
   event or abrupt failure).

### `correlation` strategy
Simpler: Pearson/Spearman correlation between `NCE_Value` and `Pre_Execute_Time`; significant
correlation above a threshold flags drift, otherwise stable.

## 9. Root_Cause_Type Taxonomy

`LITHO_TOOL_ISSUE`, `LITHO_CHUCK_ISSUE`, `LITHO_CHUCK_CONTAMINATION` (from noise filter — never
reach root-cause/drift stages), `SPECIFIC_CHAMBER_DEFECT` (static, no time trend),
`CHAMBER_DRIFT` (gradual worsening), `CHAMBER_SUDDEN_SHIFT` (CUSUM-detected change point).

**Out of scope for v1**: `PRODUCT_TOOL_MISMATCH` (PartID × Tool/Chamber interaction). Explicitly
deferred — would require adding `PartID` as an analysis dimension in the root-cause contingency
table / ML features, and needs larger sample sizes to be statistically stable. Candidate for v2.

## 10. Schemas

### Config
```python
class AnalysisConfig(BaseModel):
    spec_threshold: float = 15.0
    min_wafer_count: int = 5
    hotspot_ratio_threshold: float = 0.05
    noise_filter_majority_threshold: float = 0.5
    root_cause_strategy: Literal["statistical", "ml", "both"] = "both"
    drift_strategy: Literal["regression_cusum", "correlation"] = "regression_cusum"
    alpha: float = 0.05
```

### Output
```python
class RootCauseDetail(BaseModel):
    Suspect_Pre_ToolID: str
    Suspect_Pre_ChamberID: str
    Confidence_Score: float  # 0-100
    Root_Cause_Type: Literal[
        "LITHO_TOOL_ISSUE", "LITHO_CHUCK_ISSUE", "LITHO_CHUCK_CONTAMINATION",
        "SPECIFIC_CHAMBER_DEFECT", "CHAMBER_DRIFT", "CHAMBER_SUDDEN_SHIFT",
    ]
    Affected_Coordinates: list[tuple[float, float]]
    Metrics: dict[str, float]
    Requires_Manual_Review: bool = False

class AnalysisResult(BaseModel):
    Summary: list[RootCauseDetail]   # sorted by Confidence_Score desc
    Details: list[RootCauseDetail]
    Generated_At: datetime
    Config_Used: AnalysisConfig
```

Entry point: `result: AnalysisResult = pipeline.run(raw_df, config)`;
`result.model_dump_json()` for JSON output.

## 11. Error Handling

Validate only at system boundaries; trust internal data once past them.

- Input: Pydantic validates required fields/types. A wafer with empty `Measurement_Points` is
  skipped with a logged warning — does not abort the batch.
- Preprocessing: if the history-level regex matches **zero** `Pre_*_i` columns, raise
  `PreprocessingError` explicitly (this means the input doesn't match the expected schema — it
  should not be silently treated as "no upstream history").
  a hotspot's total wafer count below `min_wafer_count` is marked `insufficient_sample` and
  reported with its raw ratio only — no statistical/ML confidence is computed, to avoid a
  spurious confidence score from too few samples.
- Root-cause statistical strategy's chi-square→Fisher fallback is a designed behavior, not an
  exception path.

## 12. Testing Plan

- Per-strategy unit tests using small synthetic DataFrames with a known ground truth (e.g.
  construct data where `CMP_01_ChamberA` causes 90% anomaly rate, verify both `statistical` and
  `ml` strategies identify it).
- Dedicated noise-filter tests for each of the three domain rules: chuck majority, tool majority,
  chuck-majority-spanning-multiple-stages (contamination).
- Dedicated preprocessing tests: dynamic N discovery, correct dropping of missing history levels,
  correct `Measurement_Points` explosion.
- One end-to-end fixture test: a full synthetic batch run through the whole pipeline, asserting
  the final JSON structure and that it identifies the known-injected root cause.
