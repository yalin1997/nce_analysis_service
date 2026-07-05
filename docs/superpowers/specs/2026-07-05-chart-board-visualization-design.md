# Chart Board Visualization — Design

Date: 2026-07-05
Status: Approved (pending user sign-off on written spec)

## 1. Purpose

Today `pipeline.run` returns an `AnalysisResult` — a flat, structured summary
of root-cause suspects — but there is no way to *see* the underlying wafer
data that led to those conclusions. An engineer reading the JSON has to trust
the confidence score; they can't visually confirm "yes, this chamber's wafers
really do trend upward over time" or "yes, this hotspot really is
concentrated at one physical location on the wafer."

This adds a single self-contained HTML report — `chart_board.html` — that
visualizes an `AnalysisResult` alongside the preprocessed wafer data it was
computed from: an SPC-style chart per top-ranked suspect (wafer `NCE_Value`
over time, grouped/colored by machine), and a spatial heatmap of anomaly
ratio across every measured wafer coordinate.

**Analysis premise this design assumes** (confirmed during brainstorming):
the pipeline measures `NCE_Value` at a single LITHO stage per wafer, and
traces anomalies back to the *upstream* (`Pre_*`) tool/chamber history — or,
when the noise filter determines the anomaly is really caused by LITHO's own
Tool/Chuck, terminates there instead. Every chart-board panel is built from
one of these two branches; there is no third case.

## 2. Scope

**In scope**: a new `nce_analysis/viz/chart_board.py` module
(`render_chart_board`), a new `root_cause/base.py` helper
(`resolve_detail_history`) that reconstructs per-suspect wafer history from
`long_df`, a `--chart-board PATH` CLI flag, a correctness fix to
`noise_filter/majority_rule.py`'s chuck-majority counting (surfaced while
designing the LITHO-self-issue chart branch — see §4), and the `plotly`
dependency.

**Out of scope**: cross-highlighting between the heatmap and the sidebar
(clicking a coordinate does not scroll to/highlight its suspect panel, and
vice versa), CUSUM/regression trend-line overlays on the SPC chart, any
configurability of chart-board appearance (no theming flags), charting every
`Details` entry (only `Summary`'s top N get a panel — see §5).

## 3. Architecture & Entry Points

`nce_analysis/viz/chart_board.py`:

```python
def render_chart_board(
    result: AnalysisResult, long_df: pd.DataFrame, config: AnalysisConfig
) -> str:
    """Render a self-contained HTML report (sidebar of ranked suspects, a
    per-suspect SPC chart, and a wafer-coordinate anomaly-ratio heatmap).
    Returns the HTML document as a string; the caller decides whether to
    write it to disk. long_df is the preprocessed (WideHistoryReshape)
    output — same shape pipeline.run consumes internally."""
```

This mirrors `pipeline.run`'s own convention: the library function returns
data, the caller (CLI or user script) decides what to do with it — no file
I/O inside `render_chart_board` itself.

**CLI** (`cli.py`): new `--chart-board PATH` argument. When passed, `main()`
additionally runs `long_df = WideHistoryReshape().transform(raw_df)` and
writes `render_chart_board(result, long_df, config)` to that path.
`--chart-board` is independent of `--output` (JSON) — either, both, or
neither may be passed in one invocation.

This means preprocessing runs twice when `--chart-board` is used (once here,
once again inside `pipeline.run`). Accepted trade-off: this repo's data scale
is explicitly "hundreds to low-thousands of wafers per call... no need to
optimize for larger data" (per `CLAUDE.md`), and avoiding it would require
changing `pipeline.run`'s return contract, which every existing caller and
test depends on today.

New dependency: `plotly`, added to `pyproject.toml`.

## 4. Data Resolution: `resolve_detail_history`

`AnalysisResult` only keeps aggregated fields (confidence, metrics, affected
coordinates) — not the underlying wafer-level rows a chart needs. Rather
than re-deriving filtering/grouping logic a third time (alongside
`pipeline.py` and `majority_rule.py`), a new function lives in
`root_cause/base.py`, next to the other suspect-matching helpers this file
already owns (`build_suspect_key`/`split_suspect_key`/`matches_suspect`):

```python
def resolve_detail_history(
    long_df: pd.DataFrame, detail: RootCauseDetail, config: AnalysisConfig
) -> pd.DataFrame:
    """Given one RootCauseDetail from AnalysisResult, reconstruct the
    wafer-level rows it was computed from. Returns columns normalized for
    chart consumption: time, NCE_Value, group_label, WaferID, is_anomaly,
    is_suspect_group."""
```

Behavior branches on `detail.Root_Cause_Type`. For the two LITHO-self types,
the **filtered (fixed) dimension** is whatever the original classification
actually keyed on; the **remaining (free) dimension** becomes `group_label`
so the chart can still show real variation instead of a meaningless
single-color group. For the upstream types, the filter is deliberately
*broader than the suspect* — see rationale below the table:

| Root_Cause_Type | Filter | `group_label` | time axis |
|---|---|---|---|
| `LITHO_CHUCK_ISSUE` / `_CONTAMINATION` | `ToolID == Suspect_Pre_ToolID` **and** `ChuckID == Suspect_Pre_ChamberID` (composite — see rationale below) | none (both dims pinned) — single group | `Execute_Time` |
| `LITHO_TOOL_ISSUE` | `ToolID == Suspect_Pre_ToolID` | `ChuckID` (chuck was never considered for this classification, `Suspect_Pre_ChamberID` is `"N/A"`) | `Execute_Time` |
| `SPECIFIC_CHAMBER_DEFECT` / `CHAMBER_DRIFT` / `CHAMBER_SUDDEN_SHIFT` | `Pre_StepID == Suspect_Pre_StepID` only — **every** Tool(+Chamber) at that process step, not just the suspect | every distinct `Pre_ToolID` (`granularity == "tool"`) or `Pre_ToolID\|Pre_ChamberID` combo (`granularity == "chamber"`) present in the group | `Pre_Execute_Time` |

All branches first filter to `X_Posi/Y_Posi ∈ detail.Affected_Coordinates`
and dedupe by `(WaferID, X_Posi, Y_Posi)` (LITHO-self branch, matching
`majority_rule.py`) or by `WaferID` keeping the latest timestamp (upstream
branch, matching `pipeline.py`'s rework handling) before computing
`is_anomaly = NCE_Value > config.spec_threshold`.

**Why the upstream branch shows every combo at that step, not just the
suspect:** the point of the chart is to visually confirm the recommended
Tool/Chamber's `NCE_Value` trend is actually worse than its peers at the
same process step — that comparison is impossible if the chart only ever
shows the suspect in isolation. `matches_suspect(long_df, candidate, config)`
still runs here, but as a **tag**, not a filter: it produces the
`is_suspect_group` boolean column so the render layer (§5) knows which rows
are the suspect's own, without re-deriving the granularity decision. The
`X_Posi/Y_Posi ∈ detail.Affected_Coordinates` restriction still applies to
every row, suspect and peers alike — the comparison is "same physical
coordinate, different upstream machine," not different coordinates
entirely, otherwise location-driven variation would be conflated with
machine-driven variation.

**Why the chuck branch filters on `ToolID` *and* `ChuckID` together, not
`ChuckID` alone:** `ChuckID` is only unique *within* a LITHO tool (each tool
has its own Chuck "1"/"2" — conceptually the same role `ChamberID` plays
upstream, just named differently). Filtering by `ChuckID` alone would pull
in every other tool's Chuck "1"/"2" too. Since `Suspect_Pre_ToolID` for these
types is already the derived owning-tool value recorded on the detail, the
composite filter is available for free.

**Bundled fix — `noise_filter/majority_rule.py`:** this same ambiguity
exists one level up, in how the suspect is *decided* in the first place.
Today `chuck_counts = anomalous["ChuckID"].value_counts()` counts raw
`ChuckID` across every tool at a hotspot, then derives `owning_tool` via
`.mode()` — so two different tools' Chuck "1" could be miscounted as one
majority. Fix: count `(ToolID, ChuckID)` pairs directly —

```python
combo_counts = anomalous.groupby(["ToolID", "ChuckID"]).size().sort_values(ascending=False)
(owning_tool, top_chuck), chuck_n = combo_counts.index[0], combo_counts.iloc[0]
chuck_share = chuck_n / total_anomalous
if chuck_share > config.noise_filter_majority_threshold:
    chuck_rows = anomalous[(anomalous["ToolID"] == owning_tool) & (anomalous["ChuckID"] == top_chuck)]
    stage_diversity = chuck_rows["StageID"].nunique()
    # owning_tool is now exact; .mode() is no longer needed
```

Verified against `tests/test_majority_rule.py`: no existing case has the
same `ChuckID` under two different `ToolID`s, so this is additive — existing
tests keep passing unchanged. A new test covers the corrected scenario (same
`ChuckID` split across two tools, neither individual (Tool, Chuck) pair
reaching majority alone).

## 5. Chart Specs

Designed via the `dataviz` skill's procedure (form → color → marks →
interaction), reusing its validated default palette unchanged.

### SPC chart (one per `Summary` entry — top `config.summary_top_n`, not
every `Details` entry)

- Scatter over time. Color's job (identity: which machine) and status
  (in-spec vs. anomalous) are different jobs, so they go on two different
  channels rather than colliding on one hue:
  - **Hue** (categorical, fixed 8-slot order: blue/aqua/yellow/green/violet/
    red/magenta/orange) = `group_label`. For LITHO-self panels, typically
    1–2 distinct values (per §4's fixed/free split). For upstream panels,
    potentially many — every Tool(+Chamber) sharing that process step.
  - **Shape** = anomaly status: circle = within spec, diamond =
    `NCE_Value > spec_threshold`.
  - Colors are assigned **once, globally, across every panel in the report**
    before rendering any individual chart — not per-panel — so the same
    real Tool/Chamber/Chuck value gets the same color everywhere it appears.
    Assignment order: **every entity that is `is_suspect_group` in at least
    one panel gets a real slot first**, decided once across the whole
    report — so a suspect can never be bumped to "Other" in its own panel
    just because it's globally low-traffic; remaining slots go to the
    highest-wafer-count non-suspect entities; anything past 8 total folds
    into a neutral "Other." "Other" keeps its true Tool/Chamber identity in
    the hover tooltip — only the color, not the data, is compressed. (With
    `summary_top_n` at its default of 5, at most 5 distinct suspects compete
    for the 8 slots, so this fits comfortably; a pathological case — more
    distinct suspect entities than color slots — is not specially handled
    beyond "some suspects also fold into Other," same as any other entity.)
- Marker ≥8px with a 2px surface-color ring.
- Horizontal dashed reference line at `config.spec_threshold`, status
  **critical** color (`#d03b3b`), directly annotated "Spec limit: {value}".
- Legend shows color groups, with the suspect's entry text-suffixed
  `"(suspect)"` (e.g. `"ToolA / ChamberX (suspect)"`) so it's identifiable
  without a dedicated color/shape channel — text carries this, not a new
  visual channel. Omitted entirely only when there's a single group (the
  panel header already names the suspect in that case). One caption under
  the chart explains the shape convention once ("○ within spec · ◆ above
  spec limit"), since that meaning is constant across every panel.
- Hover tooltip: `WaferID` and `NCE_Value` lead, then time, group, status.
- No trend/CUSUM overlay.

### Wafer heatmap (spatial anomaly-ratio map, background = every measured
coordinate, not just flagged hotspots)

- Sequential single-hue (blue, steps 100→700) mapped continuously to
  `anomaly_ratio` (dedupe by `(WaferID, X_Posi, Y_Posi)`, same formula
  `hotspot/ratio_threshold.py` uses) for **every** coordinate in `long_df`,
  with a colorbar legend.
- Coordinates that were actually flagged — present in either
  `Insufficient_Sample_Hotspots` or `Details` — render as a **square**
  instead of the background **circle** (magnitude-color × flagged-status
  shape, same composite-encoding move as the SPC chart).
- Equal-aspect axes (no assumed wafer-diameter boundary circle — the data
  doesn't carry a known wafer radius).
- Hover: coordinate, ratio, wafer counts, and — if flagged — the resolved
  outcome text (`Root_Cause_Type`, or `"insufficient_sample"`).
- A "View as table" toggle listing the flagged coordinates as rows (the
  dataviz skill's requirement: every continuous color scale needs a
  non-color-only twin).

## 6. Page Layout

Single-file HTML, plotly.js embedded inline once (not once per chart, to
avoid duplicating the ~4.5 MB library across N panels) — the whole file
works fully offline.

- **Header**: title, `Generated_At`, and a one-line config summary
  (`spec_threshold`, `root_cause_granularity`, `root_cause_strategy`,
  `drift_strategy`) so the file is self-describing without the original
  JSON alongside it.
- **Sidebar** (ranked list, one row per `Summary` entry): rank, suspect label
  (`{Pre_ToolID} / {Pre_ChamberID}` for upstream types; for LITHO-self types
  `{ToolID} / Chuck {ChamberID}`, or just `{ToolID}` when `Suspect_Pre_ChamberID
  == "N/A"` — `LITHO_TOOL_ISSUE`, and upstream `granularity == "tool"`, both
  omit the Chuck/Chamber suffix rather than rendering the literal `"N/A"`),
  `Root_Cause_Type` as a plain muted badge (a diagnosis category, not a
  good/bad state — no status color), `Confidence_Score`,
  and a **warning**-status badge ("⚠ Needs manual review") when
  `Requires_Manual_Review` is true. Footer note: "Showing top {N} of
  {total}" whenever `Details` has more entries than made `Summary`.
- **Detail panel** (populated on row click, defaults to rank #1 on load):
  suspect header repeated, `Metrics` dict as a plain key/value table (no
  per-metric special-casing — covers chi-square/Fisher, SHAP, chuck/tool
  share, drift/CUSUM stats, or the `insufficient_*` sentinels uniformly),
  `Affected_Coordinates` as plain text/small table (no cross-highlight into
  the heatmap — cut for v1), and the SPC chart.
- **Wafer measurement map** (below, full width): the heatmap plus its table
  toggle.
- **Empty states**: no `Summary` entries → sidebar and detail panel show a
  placeholder message; heatmap still renders from `long_df` background
  alone.

## 7. Error Handling

Consistent with the project's existing rule (validate only at boundaries,
trust internal data past them). `render_chart_board` runs after
`pipeline.run` has already succeeded, so batch-level validation
(`PreprocessingError` cases) has already happened upstream.

- `resolve_detail_history` returning 0 rows for a detail (shouldn't happen
  given `result`/`long_df` come from the same run, but not structurally
  impossible for a misused/mismatched pair): that one panel renders a text
  note ("no matching wafer data found") instead of a chart, rather than
  failing the whole report.
- Missing/NaN `group_label` values (partial history level): `fillna
  ("UNKNOWN")` before grouping — same pattern `pipeline.py` already uses for
  `Pre_StepID`.
- `--chart-board` write failures (bad path, unwritable dir): no new
  try/except — same convention `--output` already uses, uncaught `OSError`
  propagates as a normal traceback.

## 8. Dependencies (`pyproject.toml`)

Add to `[project.dependencies]`: `plotly`.

## 9. Testing Plan

- `tests/test_root_cause_base.py`: `resolve_detail_history` for all three
  branches (chuck-composite filter, tool-with-free-chuck, upstream
  tool-vs-chamber granularity), the rework-dedup path, the missing
  `group_label` → `"UNKNOWN"` fallback, and — for the upstream branch —
  that rows from a non-suspect Tool/Chamber at the same `Pre_StepID` are
  included with `is_suspect_group == False` while the suspect's own rows
  have `is_suspect_group == True`.
- `tests/test_majority_rule.py`: new case for the `(ToolID, ChuckID)`
  composite-counting fix (same `ChuckID` split across two tools).
- `tests/test_chart_board.py` (new): `render_chart_board` against a small
  synthetic `AnalysisResult` + `long_df` — asserts the returned string
  contains one sidebar entry per `Summary` item (not per `Details` item),
  the `spec_threshold` value, plotly.js embedded exactly once, doesn't
  crash on an empty `Summary`, the suspect's `group_label` always lands in
  a real color slot (never folded into "Other") when more than 8 combos
  are present at one step, and the suspect's legend entry carries the
  `"(suspect)"` suffix.
- `tests/test_cli.py`: `--chart-board` writes a non-empty HTML file;
  combinable with `--output` in the same invocation.
