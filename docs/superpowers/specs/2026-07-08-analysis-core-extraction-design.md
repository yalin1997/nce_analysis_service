# Analysis Core Extraction — Design

Date: 2026-07-08
Status: Approved

## 1. Goal

Extract the domain-independent analytical algorithms currently embedded in the
NCE pipeline's strategy classes into a reusable, generic package
(`analysis_core`) with neutral interfaces and per-component typed configs — so
the same statistical building blocks can be reused by future analysis
services without dragging NCE/wafer domain concepts along.

This is a **behavior-preserving refactor**: `nce_analysis.pipeline.run` must
produce byte-identical output before and after, and the existing test suite
must pass unmodified throughout.

## 2. Decisions taken during brainstorming

| Question | Decision |
|---|---|
| Where do reusable components live? | Same repo, new top-level package `analysis_core/` parallel to `nce_analysis/` (extract to its own distribution later only when a second consumer exists) |
| Interface abstraction level | Neutral column contract: DataFrame in, column names supplied via config, neutral result objects out |
| Which algorithms move? | All five analysis algorithm classes: statistical association, risk uplift, trend/change-point (both variants), majority dominance, ratio-threshold hotspot |
| Preprocessing? | **Not** extracted — `WideHistoryReshape` / `explode_measurement_points` stay domain-owned as-is |
| Config style | One pydantic config per component (algorithm params + column-name mapping); no registry/factory layer |

## 3. Package layout and dependency rule

```
nce_analysis_service/
├── analysis_core/            # generic layer (new)
│   ├── __init__.py           # re-exports all public components and configs
│   ├── stats.py              # holm_bonferroni_adjust (pure functions)
│   ├── association.py        # ChiSquareFisherRanker + AssociationConfig + GroupAssociation
│   ├── uplift.py             # RiskUpliftRanker + UpliftConfig + GroupUplift
│   ├── trend.py              # RegressionCusumDetector, CorrelationTrendDetector + TrendConfig + TrendResult
│   ├── dominance.py          # find_dominant + DominanceConfig + Dominance
│   └── ratio_hotspot.py      # RatioHotspotDetector + RatioHotspotConfig
└── nce_analysis/             # domain layer (existing; strategies become thin adapters)
```

**One-way dependency, enforced by a test:** `analysis_core` may import only
numpy / pandas / scipy / pydantic (and stdlib). It must never import
`nce_analysis` and must contain no domain vocabulary (wafer, tool, chamber,
LITHO, NCE) in its public interface. A unit test scans `analysis_core/`
sources and fails on any `nce_analysis` import. `nce_analysis` strategies
depend on `analysis_core`, never the reverse.

Packaging: add `analysis_core` to
`[tool.hatch.build.targets.wheel] packages = ["nce_analysis", "analysis_core"]`.
Future extraction into its own distribution is then a directory move plus a
pyproject edit — no import-path churn inside `analysis_core` itself.

## 4. Generic component interfaces

Shared conventions:

- Each component takes exactly one pydantic config (`extra="forbid"`)
  carrying **algorithm parameters and column-name mappings**.
- Input is any `pd.DataFrame`. On call, the component validates that every
  configured column exists, raising `ValueError` naming the missing columns.
- Outputs are neutral dataclasses / DataFrames with no domain vocabulary.
  Numeric details go in `metrics: dict[str, float]`; adapters pick and rename.

### 4.1 `association.py` — from `StatisticalStrategy`

```python
class AssociationConfig(BaseModel):
    group_col: str = "group"
    event_col: str = "is_event"        # boolean column
    alpha: float = 0.05
    min_expected_freq: float = 5.0     # chi-square expected-count gate

@dataclass
class GroupAssociation:
    group: str
    p_value: float                     # raw one-vs-rest Fisher p
    p_value_adjusted: float            # Holm-Bonferroni adjusted
    odds_ratio: float
    metrics: dict[str, float]          # num_hypotheses, sample_size,
                                       # fisher_fallback, p_value_global (when computed)

class ChiSquareFisherRanker:
    def __init__(self, config: AssociationConfig): ...
    def rank(self, df: pd.DataFrame) -> list[GroupAssociation]: ...
```

Algorithm (moved verbatim from `StatisticalStrategy.analyze`): crosstab of
`group_col` × `event_col`; fewer than 2 groups → `[]`. Compute expected
frequencies; if any expected cell < `min_expected_freq`, skip the global
chi-square gate (fisher fallback), else run chi-square and return `[]` when
global p ≥ alpha. Per-group one-vs-rest Fisher exact (`alternative="greater"`),
Holm-Bonferroni adjust across all groups tested. Return only groups with
`odds_ratio > 1` and `p_value_adjusted < alpha`, sorted by `p_value_adjusted`
ascending. Empty list = no significant association.

`_holm_bonferroni_adjust` moves to `analysis_core/stats.py` as public
`holm_bonferroni_adjust`.

### 4.2 `uplift.py` — from `MLStrategy`

```python
class UpliftConfig(BaseModel):
    group_col: str = "group"
    event_col: str = "is_event"

@dataclass
class GroupUplift:
    group: str
    uplift: float                      # group event rate − overall event rate
    normalized_share: float            # uplift / sum of all positive uplifts
    metrics: dict[str, float]          # group_event_rate, overall_event_rate,
                                       # group_event_count, group_size, sample_size

class RiskUpliftRanker:
    def __init__(self, config: UpliftConfig): ...
    def rank(self, df: pd.DataFrame) -> list[GroupUplift]: ...
```

Returns `[]` when fewer than 2 groups, or labels are all-true/all-false, or no
group has positive uplift. Otherwise returns positive-uplift groups sorted by
`uplift` descending.

### 4.3 `trend.py` — from `RegressionCusum` and `Correlation`

```python
class TrendConfig(BaseModel):
    time_col: str = "timestamp"
    value_col: str = "value"
    alpha: float = 0.05
    cusum_sigma: float = 3.0           # previously hard-coded 3.0

@dataclass
class TrendResult:
    kind: Literal["change_point", "increasing_trend", "none"]
    metrics: dict[str, float]

class RegressionCusumDetector:
    def __init__(self, config: TrendConfig): ...
    def detect(self, df: pd.DataFrame) -> TrendResult: ...

class CorrelationTrendDetector:
    def __init__(self, config: TrendConfig): ...
    def detect(self, df: pd.DataFrame) -> TrendResult: ...
```

Both detectors own the time normalization: sort by `time_col`, parse with
`pd.to_datetime`, convert to elapsed hours from the first timestamp.
Degenerate series (fewer than 2 distinct times, or for the correlation
variant fewer than 2 distinct values) return `kind="none"` with an
`insufficient_time_variation` / `insufficient_value_variation` metric — no
exception. RegressionCusum keeps the sample-size-scaled CUSUM threshold
(`cusum_sigma * residual_std * sqrt(n)`) and its `cusum_threshold_method`
metric. `change_point` outranks `increasing_trend` exactly as today.

### 4.4 `dominance.py` — kernel of `MajorityRule`

```python
class DominanceConfig(BaseModel):
    threshold: float = 0.5             # share must be strictly greater

@dataclass
class Dominance:
    category: tuple                    # supports composite keys, e.g. (tool, chuck)
    share: float
    count: int

def find_dominant(keys: pd.Series, config: DominanceConfig) -> Dominance | None: ...
```

Only the "does one category strictly exceed the share threshold" kernel is
generic. `MajorityRule`'s two-tier check (Tool+Chuck combo first, then Tool),
stage-diversity contamination rule, and LITHO_* labels are domain logic and
stay in the adapter. Extracting more would create a fake-generic interface
with a single consumer.

`find_dominant` on an empty series returns `None`. Single-element categories
(plain strings) are wrapped as 1-tuples for a uniform return shape.

### 4.5 `ratio_hotspot.py` — from `RatioThreshold`

```python
class RatioHotspotConfig(BaseModel):
    group_cols: list[str]              # e.g. ["X_Posi", "Y_Posi"]
    event_col: str = "is_event"
    ratio_threshold: float = 0.05
    min_count: int = 5

class RatioHotspotDetector:
    def __init__(self, config: RatioHotspotConfig): ...
    def detect(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

Output columns: the `group_cols`, plus `event_count`, `total_count`,
`event_ratio`, `insufficient_sample` (bool), filtered to
`event_ratio >= ratio_threshold`, sorted by `(event_ratio, event_count)`
descending. Deduplication of input rows is the caller's job (it is
domain-specific: LITHO point identity).

### 4.6 Deliberate interface choices

1. **Rankers return the full sorted eligible list, not a single winner.**
   "Pick the best" is a domain decision; adapters take `[0]`. Lists are more
   reusable for future consumers (top-k reporting, cross-validation).
2. **Confidence-score formulas stay in adapters.** `(1 − raw_p) × 100` and
   `normalized_share × 100` are NCE reporting conventions, not statistics.

## 5. NCE adapter changes

Class names, stage ABCs, `AnalysisConfig`, and `pipeline.py` call sites are
all unchanged. Each concrete strategy becomes a thin adapter:

- **`root_cause/statistical.py`** — `build_suspect_key` → `ChiSquareFisherRanker(
  AssociationConfig(group_col="suspect_key", event_col="is_anomaly",
  alpha=config.alpha)).rank(...)` → take the first result, `split_suspect_key`,
  compute `(1 − p_value) × 100`, and map metrics back to the existing keys
  (`p_value_combo`, `p_value_combo_adjusted`, `num_hypotheses`,
  `multiple_testing_method`, `odds_ratio`, `sample_size`, `fisher_fallback`,
  `p_value_global`). Output must be byte-identical.
- **`root_cause/ml.py`** — same pattern via `RiskUpliftRanker`;
  `normalized_share × 100` is the confidence score; existing metric keys
  (`risk_uplift`, `suspect_anomaly_rate`, `overall_anomaly_rate`,
  `suspect_anomaly_count`, `suspect_sample_size`, `sample_size`,
  `ml_scoring_method`) preserved.
- **`drift/regression_cusum.py`, `drift/correlation.py`** — construct the
  matching detector with `time_col="Pre_Execute_Time"`,
  `value_col="NCE_Value"`, `alpha=config.alpha`; map `TrendResult.kind`:
  `change_point → CHAMBER_SUDDEN_SHIFT`, `increasing_trend → CHAMBER_DRIFT`,
  `none → SPECIFIC_CHAMBER_DEFECT`. Metric keys pass through unchanged.
- **`hotspot/ratio_threshold.py`** — adapter keeps the LITHO point dedup and
  `is_anomaly` derivation (`NCE_Value > spec_threshold`), calls
  `RatioHotspotDetector`, then renames `event_count/total_count/event_ratio`
  back to `anomalous_wafer_count/total_wafer_count/anomaly_ratio` so every
  downstream consumer sees the exact current schema.
- **`noise_filter/majority_rule.py`** — the two dominance checks become
  `find_dominant` calls (first on the `(ToolID, ChuckID)` composite key, then
  on `ToolID`); stage-diversity classification and all LITHO_* labeling stay
  in place.

## 6. Error handling

- Generic components validate their own boundary only: missing configured
  columns raise `ValueError` naming them.
- Degenerate inputs that today return early (constant series, empty groups,
  all-same labels) keep returning empty/neutral results rather than raising,
  so adapters need no try/except.
- Domain validation (empty `Measurement_Points`, missing history columns,
  duplicate-wafer handling) is untouched and stays in `nce_analysis`.

## 7. Testing

1. **Existing suite is the behavior-preservation net**: no existing test file
   is modified, and the full suite must be green after every migration step.
2. **New unit tests per component** under `tests/analysis_core/`, using
   neutral fixtures (no wafer/tool vocabulary). Coverage per component:
   happy path, empty-result paths, Fisher fallback trigger, Holm adjustment
   ordering, degenerate time/value series, composite dominance keys,
   insufficient-sample flagging, and missing-column `ValueError`.
3. **Dependency-direction test**: scans `analysis_core/` source files and
   fails on any `nce_analysis` import.

## 8. Migration order (input to the implementation plan)

TDD throughout; full suite run after each step:

1. Create `analysis_core` package + `stats.py` (`holm_bonferroni_adjust`)
   with tests; update pyproject packaging; add the dependency-direction test.
2. `association.py` (+ tests) → rewire `root_cause/statistical.py`.
3. `uplift.py` (+ tests) → rewire `root_cause/ml.py`.
4. `trend.py` (+ tests) → rewire both drift strategies.
5. `ratio_hotspot.py` (+ tests) → rewire `hotspot/ratio_threshold.py`.
6. `dominance.py` (+ tests) → rewire `noise_filter/majority_rule.py`.
7. Update `CLAUDE.md` Architecture section and `README.md` to describe the
   two-layer structure.

## 9. Out of scope

- Extracting preprocessing (`WideHistoryReshape`, `explode_measurement_points`).
- A component registry / config-driven pipeline assembly.
- Publishing `analysis_core` as a separate distribution (revisit when a
  second consumer exists).
- Any behavior change, new algorithm, or tuning of thresholds/formulas.
