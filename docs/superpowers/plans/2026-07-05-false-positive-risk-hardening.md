# False Positive Risk Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce known false-positive and disagreement risks in drift, statistical root-cause, and ML root-cause analysis without expanding v1 scope into `PartID` / product-tool mismatch analysis.

**Architecture:** Keep the existing strategy interfaces and pipeline orchestration intact. Harden `RegressionCusum` with a sample-size-aware sudden-shift detector, harden `StatisticalStrategy` with post-hoc p-value correction, and replace the current suspect-key-only SHAP decision tree with a simpler suspect uplift scorer that preserves the `MLStrategy` public contract.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, scikit-learn only where still used elsewhere, pytest. Do not add new runtime dependencies.

---

## Scope Rules

- Do not add `PartID` to root-cause features or statistical grouping. `PRODUCT_TOOL_MISMATCH` remains out of scope for v1.
- Do not add `ChuckID` to upstream root-cause ML. LITHO `ChuckID` remains part of `MajorityRule` noise filtering.
- Preserve these public entry points: `nce_analysis.pipeline.run`, `RegressionCusum.classify`, `StatisticalStrategy.analyze`, `MLStrategy.analyze`, `BothStrategy.analyze`.
- Preserve `AnalysisConfig.root_cause_granularity` behavior by continuing to use `build_suspect_key` and `split_suspect_key`.
- Use the repo command style from `CLAUDE.md`: `python3.12 -m pytest ...`.

## File Structure

- Modify: `nce_analysis/drift/regression_cusum.py`
  - Responsible for linear drift and sudden-shift classification.
  - Add a sample-size-aware CUSUM threshold or standard two-sided SPC CUSUM.
- Modify: `tests/test_regression_cusum.py`
  - Add deterministic regression tests for pure-noise false positives and real sudden shifts.
- Modify: `nce_analysis/root_cause/statistical.py`
  - Responsible for chi-square gating and per-suspect Fisher exact tests.
  - Add Holm-Bonferroni adjusted p-values before accepting a candidate.
- Modify: `tests/test_statistical.py`
  - Add tests proving raw-significant but Holm-insignificant suspects are rejected, while strong suspects survive.
- Modify: `nce_analysis/root_cause/ml.py`
  - Responsible for ML-style root-cause candidates.
  - Replace suspect-key-only decision tree + SHAP with a direct, interpretable suspect risk-uplift scorer.
- Modify: `tests/test_ml.py`
  - Add high-cardinality tests proving all suspect keys remain eligible and no tree-depth resolution limit exists.
- Optional modify: `pyproject.toml`
  - Remove `shap` dependency only after confirming no other code imports it.

---

## Task 1: Harden CUSUM Against Pure-Noise False Positives

**Files:**
- Modify: `tests/test_regression_cusum.py`
- Modify: `nce_analysis/drift/regression_cusum.py`

- [ ] **Step 1: Write the failing pure-noise regression test**

Append this test to `tests/test_regression_cusum.py`:

```python
def test_classify_does_not_report_sudden_shift_for_seeded_white_noise():
    rng = np.random.default_rng(0)
    n = 50
    values = 20.0 + rng.normal(0.0, 1.0, n)
    series_df = pd.DataFrame(
        {"Pre_Execute_Time": _timestamps(n), "NCE_Value": values}
    )
    config = AnalysisConfig(alpha=0.05)

    root_cause_type, metrics = RegressionCusum().classify(series_df, config)

    assert root_cause_type == "SPECIFIC_CHAMBER_DEFECT"
    assert metrics["cusum_threshold"] > metrics["cusum_range"]
    assert metrics["cusum_threshold_method"] == 1.0
```

Also add `import numpy as np` at the top of `tests/test_regression_cusum.py` if it is not already present.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python3.12 -m pytest tests/test_regression_cusum.py::test_classify_does_not_report_sudden_shift_for_seeded_white_noise -v
```

Expected: FAIL because the current fixed threshold `5 * residual_std` reports `CHAMBER_SUDDEN_SHIFT` for this seeded noise series, or because `cusum_threshold_method` is missing.

- [ ] **Step 3: Implement a sample-size-aware CUSUM threshold**

In `nce_analysis/drift/regression_cusum.py`, replace:

```python
cusum_threshold = 5 * residual_std if residual_std > 0 else 0.0
change_point_detected = cusum_threshold > 0 and cusum_range > cusum_threshold
```

with:

```python
n = len(residuals)
cusum_threshold = (
    3.0 * residual_std * np.sqrt(n) if residual_std > 0 and n > 0 else 0.0
)
change_point_detected = cusum_threshold > 0 and cusum_range > cusum_threshold
```

Add this metric to the existing `metrics` dict:

```python
"cusum_threshold_method": 1.0,
```

Meaning of `cusum_threshold_method`: `1.0` means sample-size-aware residual CUSUM threshold. This keeps metrics numeric and compatible with `dict[str, float]`.

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
python3.12 -m pytest tests/test_regression_cusum.py::test_classify_does_not_report_sudden_shift_for_seeded_white_noise -v
```

Expected: PASS.

- [ ] **Step 5: Verify existing drift behavior still detects a clear sudden shift**

Run:

```bash
python3.12 -m pytest tests/test_regression_cusum.py::test_classify_detects_sudden_shift -v
```

Expected: PASS. If it fails, strengthen the existing fixture from `[10.0] * 10 + [30.0] * 10` to `[10.0] * 10 + [50.0] * 10`; the test should represent an unambiguous sudden shift under the stricter threshold.

- [ ] **Step 6: Run all drift tests**

Run:

```bash
python3.12 -m pytest tests/test_regression_cusum.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add nce_analysis/drift/regression_cusum.py tests/test_regression_cusum.py
git commit -m "fix: scale cusum threshold with sample size"
```

---

## Task 2: Add Statistical Multiple-Comparison Correction

**Files:**
- Modify: `tests/test_statistical.py`
- Modify: `nce_analysis/root_cause/statistical.py`

- [ ] **Step 1: Write the failing Holm-Bonferroni rejection test**

Append this test to `tests/test_statistical.py`:

```python
def test_analyze_rejects_raw_significant_combo_after_holm_correction():
    rows = []
    for combo_idx in range(20):
        anomaly_count = 8 if combo_idx == 0 else 3
        for i in range(20):
            rows.append(
                {
                    "Pre_ToolID": f"CMP_{combo_idx:02d}",
                    "Pre_ChamberID": f"Chamber{combo_idx:02d}",
                    "is_anomaly": i < anomaly_count,
                }
            )
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig(alpha=0.05)

    result = StatisticalStrategy().analyze(group_df, config)

    assert result == []
```

This fixture has one elevated group with a raw one-vs-rest Fisher p-value below `0.05`, but it is not strong enough to survive Holm correction across 20 suspect tests.

- [ ] **Step 2: Write the strong-signal survival test**

Append this test to `tests/test_statistical.py`:

```python
def test_analyze_keeps_strong_combo_after_holm_correction():
    rows = []
    for combo_idx in range(20):
        anomaly_count = 18 if combo_idx == 0 else 2
        for i in range(20):
            rows.append(
                {
                    "Pre_ToolID": f"CMP_{combo_idx:02d}",
                    "Pre_ChamberID": f"Chamber{combo_idx:02d}",
                    "is_anomaly": i < anomaly_count,
                }
            )
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig(alpha=0.05)

    result = StatisticalStrategy().analyze(group_df, config)

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_00"
    assert result[0].suspect_chamber_id == "Chamber00"
    assert result[0].metrics["p_value_combo_adjusted"] < config.alpha
    assert result[0].metrics["multiple_testing_method"] == 1.0
```

- [ ] **Step 3: Run the new tests and verify at least one fails**

Run:

```bash
python3.12 -m pytest tests/test_statistical.py::test_analyze_rejects_raw_significant_combo_after_holm_correction tests/test_statistical.py::test_analyze_keeps_strong_combo_after_holm_correction -v
```

Expected: FAIL because `StatisticalStrategy` currently accepts raw p-values and does not emit `p_value_combo_adjusted` or `multiple_testing_method`.

- [ ] **Step 4: Implement Holm-Bonferroni correction helper**

Add this helper near the top of `nce_analysis/root_cause/statistical.py`, below imports:

```python
def _holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in original order."""
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted_sorted: list[tuple[int, float]] = []
    running_max = 0.0
    for rank, (original_idx, p_value) in enumerate(indexed):
        adjusted = min((m - rank) * p_value, 1.0)
        running_max = max(running_max, adjusted)
        adjusted_sorted.append((original_idx, running_max))

    adjusted_by_original = [1.0] * m
    for original_idx, adjusted in adjusted_sorted:
        adjusted_by_original[original_idx] = adjusted
    return adjusted_by_original
```

- [ ] **Step 5: Use adjusted p-values to select the statistical candidate**

In `StatisticalStrategy.analyze`, replace the loop that updates `best_p` directly with collection plus adjustment:

```python
test_results = []
for suspect_key in contingency.index:
    suspect_anomaly = contingency.loc[suspect_key, True]
    suspect_normal = contingency.loc[suspect_key, False]
    rest_anomaly = overall_anomaly_count - suspect_anomaly
    rest_normal = overall_normal_count - suspect_normal
    table = [[suspect_anomaly, suspect_normal], [rest_anomaly, rest_normal]]
    odds_ratio, p_value = fisher_exact(table, alternative="greater")
    test_results.append(
        {
            "suspect_key": suspect_key,
            "p_value": float(p_value),
            "odds_ratio": float(odds_ratio),
        }
    )

adjusted_p_values = _holm_bonferroni_adjust(
    [item["p_value"] for item in test_results]
)
for item, adjusted_p in zip(test_results, adjusted_p_values, strict=True):
    item["p_value_adjusted"] = adjusted_p

eligible = [
    item
    for item in test_results
    if item["odds_ratio"] > 1 and item["p_value_adjusted"] < config.alpha
]
if not eligible:
    return []

best = min(eligible, key=lambda item: item["p_value_adjusted"])
best_suspect_key = best["suspect_key"]
best_p = best["p_value"]
best_p_adjusted = best["p_value_adjusted"]
best_odds_ratio = best["odds_ratio"]
```

Then change the old rejection check:

```python
if best_suspect_key is None or best_p >= config.alpha:
    return []
```

to remove it entirely, because `eligible` already handles rejection.

Add these metrics:

```python
"p_value_combo_adjusted": float(best_p_adjusted),
"num_hypotheses": float(len(test_results)),
"multiple_testing_method": 1.0,
```

Keep the existing `"p_value_combo"` as the raw p-value for diagnostics.

- [ ] **Step 6: Run the statistical tests**

Run:

```bash
python3.12 -m pytest tests/test_statistical.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add nce_analysis/root_cause/statistical.py tests/test_statistical.py
git commit -m "fix: apply holm correction to fisher post hoc tests"
```

---

## Task 3: Replace Suspect-Key-Only SHAP Tree With Direct Risk Uplift Scoring

**Files:**
- Modify: `tests/test_ml.py`
- Modify: `nce_analysis/root_cause/ml.py`
- Optional modify: `pyproject.toml`

- [ ] **Step 1: Write the high-cardinality ML test**

Append this test to `tests/test_ml.py`:

```python
def test_analyze_identifies_best_high_cardinality_combo_without_depth_limit():
    rows = []
    for combo_idx in range(50):
        anomaly_count = 9 if combo_idx == 17 else 1
        for i in range(10):
            rows.append(
                {
                    "Pre_ToolID": f"CMP_{combo_idx:02d}",
                    "Pre_ChamberID": f"Chamber{combo_idx:02d}",
                    "is_anomaly": i < anomaly_count,
                }
            )
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig()

    result = MLStrategy().analyze(group_df, config)

    assert len(result) == 1
    assert result[0].suspect_tool_id == "CMP_17"
    assert result[0].suspect_chamber_id == "Chamber17"
    assert result[0].metrics["risk_uplift"] > 0
    assert result[0].metrics["ml_scoring_method"] == 1.0
```

- [ ] **Step 2: Write the no-uplift empty-result test**

Append this test to `tests/test_ml.py`:

```python
def test_analyze_returns_empty_when_no_combo_has_positive_risk_uplift():
    rows = []
    for combo_idx in range(5):
        for i in range(20):
            rows.append(
                {
                    "Pre_ToolID": f"CMP_{combo_idx:02d}",
                    "Pre_ChamberID": f"Chamber{combo_idx:02d}",
                    "is_anomaly": i < 5,
                }
            )
    group_df = pd.DataFrame(rows)
    config = AnalysisConfig()

    result = MLStrategy().analyze(group_df, config)

    assert result == []
```

- [ ] **Step 3: Run the new ML tests and verify they fail or expose old metrics**

Run:

```bash
python3.12 -m pytest tests/test_ml.py::test_analyze_identifies_best_high_cardinality_combo_without_depth_limit tests/test_ml.py::test_analyze_returns_empty_when_no_combo_has_positive_risk_uplift -v
```

Expected: FAIL because the current implementation uses `DecisionTreeClassifier(max_depth=3)` and SHAP metrics instead of direct risk-uplift metrics.

- [ ] **Step 4: Replace `MLStrategy.analyze` implementation**

In `nce_analysis/root_cause/ml.py`, remove these imports:

```python
import shap
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
```

Keep `import pandas as pd`.

Replace the body after the single-suspect guard with direct uplift scoring:

```python
labels = working["is_anomaly"].astype(int)
if labels.sum() == 0 or labels.sum() == len(labels):
    return []

overall_rate = float(labels.mean())
grouped = (
    working.assign(_label=labels)
    .groupby("suspect_key")["_label"]
    .agg(["mean", "sum", "count"])
)
grouped["risk_uplift"] = grouped["mean"] - overall_rate
positive = grouped[grouped["risk_uplift"] > 0].copy()
if positive.empty:
    return []

best_suspect_key = positive["risk_uplift"].idxmax()
best = positive.loc[best_suspect_key]
positive_total = float(positive["risk_uplift"].sum())
confidence_score = (
    float(best["risk_uplift"]) / positive_total * 100.0
    if positive_total > 0
    else 0.0
)

tool_id, chamber_id = split_suspect_key(best_suspect_key, config)
return [
    RootCauseCandidate(
        suspect_tool_id=tool_id,
        suspect_chamber_id=chamber_id,
        confidence_score=float(confidence_score),
        metrics={
            "risk_uplift": float(best["risk_uplift"]),
            "suspect_anomaly_rate": float(best["mean"]),
            "overall_anomaly_rate": overall_rate,
            "suspect_anomaly_count": float(best["sum"]),
            "suspect_sample_size": float(best["count"]),
            "sample_size": float(len(working)),
            "ml_scoring_method": 1.0,
        },
    )
]
```

This keeps `MLStrategy` deterministic, interpretable, and free of tree-depth resolution limits for the current suspect-key-only feature set.

- [ ] **Step 5: Remove unused dependency only if no imports remain**

Run:

```bash
rg -n "import shap|from shap|shap\\." nce_analysis tests
```

Expected: no output. If this command has no output, remove `"shap>=0.44"` from `pyproject.toml`. If it finds a remaining legitimate import, do not edit `pyproject.toml`.

- [ ] **Step 6: Run the ML tests**

Run:

```bash
python3.12 -m pytest tests/test_ml.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

If `pyproject.toml` changed:

```bash
git add nce_analysis/root_cause/ml.py tests/test_ml.py pyproject.toml
git commit -m "fix: replace suspect-key ml tree with risk uplift scoring"
```

If `pyproject.toml` did not change:

```bash
git add nce_analysis/root_cause/ml.py tests/test_ml.py
git commit -m "fix: replace suspect-key ml tree with risk uplift scoring"
```

---

## Task 4: Verify BothStrategy and Pipeline Compatibility

**Files:**
- Modify only if tests expose an incompatibility:
  - `nce_analysis/root_cause/both.py`
  - `tests/test_both.py`
  - `tests/test_pipeline_e2e.py`

- [ ] **Step 1: Run compatibility tests**

Run:

```bash
python3.12 -m pytest tests/test_both.py tests/test_pipeline_e2e.py -v
```

Expected: PASS. The public `RootCauseCandidate` contract is unchanged, so no compatibility edit should be required.

- [ ] **Step 2: If `BothStrategy` metric assertions fail, update test expectations only**

If tests expected `"shap_contribution"` in ML metrics, replace that expectation with `"risk_uplift"`.

Example assertion:

```python
assert "ml_risk_uplift" in result[0].metrics
```

Do not change `BothStrategy` behavior unless the failure proves it cannot merge prefixed metrics from the new ML candidate.

- [ ] **Step 3: Run compatibility tests again**

Run:

```bash
python3.12 -m pytest tests/test_both.py tests/test_pipeline_e2e.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit only if files changed**

If this task changed tests:

```bash
git add tests/test_both.py tests/test_pipeline_e2e.py
git commit -m "test: update strategy compatibility expectations"
```

If no files changed, skip this commit.

---

## Task 5: Full Regression Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run targeted hardened-area tests**

Run:

```bash
python3.12 -m pytest tests/test_regression_cusum.py tests/test_statistical.py tests/test_ml.py tests/test_both.py tests/test_pipeline_e2e.py -v
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```bash
python3.12 -m pytest tests/ -v
```

Expected: PASS.

- [ ] **Step 3: Inspect for accidental v1 scope expansion**

Run:

```bash
rg -n "PartID|PRODUCT_TOOL_MISMATCH|ChuckID" nce_analysis/root_cause nce_analysis/pipeline.py tests/test_ml.py tests/test_statistical.py
```

Expected:
- No `PRODUCT_TOOL_MISMATCH` references in production code.
- No `PartID` references in `nce_analysis/root_cause`.
- No new `ChuckID` feature usage in `nce_analysis/root_cause/ml.py`.
- Existing `ChuckID` use outside root-cause noise filtering is acceptable only if it already existed before this plan.

- [ ] **Step 4: Commit any final cleanup**

If no files changed after Task 4, skip this step. If formatting or dependency cleanup changed files:

```bash
git add <changed-files>
git commit -m "chore: finalize false positive hardening"
```

---

## Self-Review Notes

**Spec coverage:** CUSUM false-positive risk is covered by Task 1. Statistical multiple-comparison risk is covered by Task 2. ML high-cardinality resolution risk is covered by Task 3. `BothStrategy` and pipeline compatibility are covered by Task 4. Full regression and v1 scope boundaries are covered by Task 5.

**No v1 scope expansion:** This plan intentionally avoids adding `PartID` and `ChuckID` to root-cause ML. It preserves the current upstream suspect key abstraction and keeps product-tool mismatch analysis out of scope.

**No placeholders:** Every task has exact files, runnable commands, expected results, and concrete code snippets for implementation.

**Type consistency:** Existing strategy signatures remain unchanged. New metrics are numeric floats to preserve `dict[str, float]` compatibility.
