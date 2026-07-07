# Analysis Core Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the domain-independent algorithms inside the NCE strategy classes into a new reusable `analysis_core` package with neutral interfaces and per-component pydantic configs, leaving the existing strategies as thin adapters.

**Architecture:** New top-level package `analysis_core/` sits parallel to `nce_analysis/` in the same repo and distribution. Strict one-way dependency: `analysis_core` imports only numpy/pandas/scipy/pydantic (enforced by a test); each existing strategy class keeps its name, ABC, and call signature but delegates its algorithm to an `analysis_core` component and maps neutral results back to domain labels/metric keys.

**Tech Stack:** Python 3.12, pandas, scipy, numpy, pydantic v2, pytest, hatchling.

Spec: `docs/superpowers/specs/2026-07-08-analysis-core-extraction-design.md`

## Global Constraints

- **Behavior-preserving refactor**: `nce_analysis.pipeline.run` output must be byte-identical; **no existing test file is modified**, and the full suite must be green after every task.
- Interpreter: always `python3.12` (a bare `python3` may resolve elsewhere). Full suite: `python3.12 -m pytest tests/ -v`.
- `analysis_core` must not import `nce_analysis` and must contain no domain vocabulary (wafer, tool, chamber, LITHO, NCE) in its public interface.
- All configs are pydantic `BaseModel` with `model_config = ConfigDict(extra="forbid")`.
- Generic components validate configured columns exist, raising `ValueError` naming the missing columns; degenerate inputs return empty/neutral results, never raise.
- TDD: write the failing test first, watch it fail, implement, watch it pass, run the full suite, commit.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Package skeleton, `stats.py`, `validation.py`, packaging, dependency-direction test

**Files:**
- Create: `analysis_core/__init__.py`
- Create: `analysis_core/stats.py`
- Create: `analysis_core/validation.py`
- Create: `tests/analysis_core/test_stats.py`
- Create: `tests/analysis_core/test_validation.py`
- Create: `tests/analysis_core/test_dependency_direction.py`
- Modify: `pyproject.toml:23-24` (wheel packages list)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `holm_bonferroni_adjust(p_values: list[float]) -> list[float]` in `analysis_core.stats`; `require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None` in `analysis_core.validation`. Every later task imports these.

- [ ] **Step 1: Write the failing tests**

Create the `tests/analysis_core/` directory **without** an `__init__.py`
(matching the existing `tests/` style — an `__init__.py` there would make
pytest import the directory as a package named `analysis_core`, colliding
with the real `analysis_core` package).

`tests/analysis_core/test_stats.py`:

```python
from analysis_core.stats import holm_bonferroni_adjust


def test_empty_input_returns_empty():
    assert holm_bonferroni_adjust([]) == []


def test_single_p_value_unchanged():
    assert holm_bonferroni_adjust([0.03]) == [0.03]


def test_adjusts_in_rank_order_with_running_max():
    # sorted: 0.01 -> 3*0.01=0.03; 0.03 -> 2*0.03=0.06; 0.04 -> max(1*0.04, 0.06)=0.06
    assert holm_bonferroni_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_caps_at_one():
    assert holm_bonferroni_adjust([0.9, 0.8]) == [1.0, 1.0]
```

`tests/analysis_core/test_validation.py`:

```python
import pandas as pd
import pytest

from analysis_core.validation import require_columns


def test_passes_when_all_columns_present():
    require_columns(pd.DataFrame({"a": [1], "b": [2]}), ["a", "b"])


def test_raises_naming_missing_columns():
    with pytest.raises(ValueError, match=r"\['b', 'c'\]"):
        require_columns(pd.DataFrame({"a": [1]}), ["a", "b", "c"])
```

`tests/analysis_core/test_dependency_direction.py`:

```python
from pathlib import Path

ANALYSIS_CORE_DIR = Path(__file__).resolve().parents[2] / "analysis_core"


def test_analysis_core_never_references_nce_analysis():
    py_files = list(ANALYSIS_CORE_DIR.rglob("*.py"))
    assert py_files, "analysis_core package not found"
    for source_file in py_files:
        content = source_file.read_text(encoding="utf-8")
        assert "nce_analysis" not in content, (
            f"{source_file} references nce_analysis; analysis_core must stay domain-free"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m pytest tests/analysis_core/ -v`
Expected: FAIL / collection errors with `ModuleNotFoundError: No module named 'analysis_core'`.

- [ ] **Step 3: Create the package**

`analysis_core/stats.py` (algorithm moved verbatim from `nce_analysis/root_cause/statistical.py::_holm_bonferroni_adjust`):

```python
def holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
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

`analysis_core/validation.py`:

```python
from collections.abc import Sequence

import pandas as pd


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise ValueError naming every configured column missing from df."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required column(s): {missing}")
```

`analysis_core/__init__.py`:

```python
from analysis_core.stats import holm_bonferroni_adjust
from analysis_core.validation import require_columns

__all__ = [
    "holm_bonferroni_adjust",
    "require_columns",
]
```

`pyproject.toml` — change the wheel packages line:

```toml
[tool.hatch.build.targets.wheel]
packages = ["nce_analysis", "analysis_core"]
```

Then reinstall the editable package so the new top-level package is registered:

Run: `python3.12 -m pip install --user --break-system-packages -e ".[dev]"`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m pytest tests/analysis_core/ -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite**

Run: `python3.12 -m pytest tests/ -v`
Expected: all PASS (existing tests untouched).

- [ ] **Step 6: Commit**

```bash
git add analysis_core/ tests/analysis_core/ pyproject.toml
git commit -m "refactor: add analysis_core package with stats and validation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `association.py` + rewire `StatisticalStrategy`

**Files:**
- Create: `analysis_core/association.py`
- Create: `tests/analysis_core/test_association.py`
- Modify: `analysis_core/__init__.py`
- Modify: `nce_analysis/root_cause/statistical.py` (full rewrite shown below)

**Interfaces:**
- Consumes: `holm_bonferroni_adjust`, `require_columns` (Task 1).
- Produces: `AssociationConfig(group_col, event_col, alpha, min_expected_freq)`, `GroupAssociation(group, p_value, p_value_adjusted, odds_ratio, metrics)`, `ChiSquareFisherRanker(config).rank(df) -> list[GroupAssociation]` — sorted by `p_value_adjusted` ascending, only groups with `odds_ratio > 1` and `p_value_adjusted < alpha`.

- [ ] **Step 1: Write the failing tests**

`tests/analysis_core/test_association.py`:

```python
import pandas as pd
import pytest

from analysis_core.association import AssociationConfig, ChiSquareFisherRanker


def make_df(rows: list[tuple[str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["group", "is_event"])


def test_flags_strongly_associated_group():
    rows = (
        [("bad", True)] * 18 + [("bad", False)] * 2
        + [("good", True)] * 2 + [("good", False)] * 18
    )
    ranked = ChiSquareFisherRanker(AssociationConfig()).rank(make_df(rows))
    assert ranked
    best = ranked[0]
    assert best.group == "bad"
    assert best.odds_ratio > 1
    assert best.p_value_adjusted < 0.05
    assert best.metrics["fisher_fallback"] == 0.0
    assert best.metrics["num_hypotheses"] == 2.0
    assert best.metrics["sample_size"] == 40.0
    assert "p_value_global" in best.metrics


def test_single_group_returns_empty():
    rows = [("only", True)] * 5 + [("only", False)] * 5
    assert ChiSquareFisherRanker(AssociationConfig()).rank(make_df(rows)) == []


def test_no_association_returns_empty():
    rows = (
        [("a", True)] * 10 + [("a", False)] * 10
        + [("b", True)] * 10 + [("b", False)] * 10
    )
    assert ChiSquareFisherRanker(AssociationConfig()).rank(make_df(rows)) == []


def test_small_expected_counts_trigger_fisher_fallback():
    rows = [("bad", True)] * 6 + [("good", True)] * 1 + [("good", False)] * 6
    ranked = ChiSquareFisherRanker(AssociationConfig()).rank(make_df(rows))
    assert ranked
    assert ranked[0].group == "bad"
    assert ranked[0].metrics["fisher_fallback"] == 1.0
    assert "p_value_global" not in ranked[0].metrics


def test_custom_column_names():
    df = pd.DataFrame(
        {
            "chamber": ["x"] * 20 + ["y"] * 20,
            "fail": [True] * 18 + [False] * 2 + [True] * 2 + [False] * 18,
        }
    )
    config = AssociationConfig(group_col="chamber", event_col="fail")
    ranked = ChiSquareFisherRanker(config).rank(df)
    assert ranked and ranked[0].group == "x"


def test_missing_column_raises():
    with pytest.raises(ValueError, match="is_event"):
        ChiSquareFisherRanker(AssociationConfig()).rank(pd.DataFrame({"group": ["a"]}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m pytest tests/analysis_core/test_association.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'analysis_core.association'`.

- [ ] **Step 3: Implement `analysis_core/association.py`**

Algorithm moved from `StatisticalStrategy.analyze`; only the vocabulary changes (suspect→group, anomaly→event) and the winner-picking moves to the caller:

```python
from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats.contingency import expected_freq

from analysis_core.stats import holm_bonferroni_adjust
from analysis_core.validation import require_columns


class AssociationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_col: str = "group"
    event_col: str = "is_event"
    alpha: float = 0.05
    min_expected_freq: float = 5.0


@dataclass
class GroupAssociation:
    group: str
    p_value: float
    p_value_adjusted: float
    odds_ratio: float
    metrics: dict[str, float] = field(default_factory=dict)


class ChiSquareFisherRanker:
    """Chi-square-gated one-vs-rest Fisher exact association ranking with
    Holm-Bonferroni correction. Returns only groups significantly enriched
    for the event (odds_ratio > 1, adjusted p < alpha), most significant
    first; empty list means no significant association."""

    def __init__(self, config: AssociationConfig):
        self.config = config

    def rank(self, df: pd.DataFrame) -> list[GroupAssociation]:
        config = self.config
        require_columns(df, [config.group_col, config.event_col])

        contingency = pd.crosstab(df[config.group_col], df[config.event_col])
        # Use reindex (not `contingency[[True, False]]`) to select/order the
        # True/False columns: pandas treats a literal [True, False] list as a
        # boolean row mask rather than column labels, which silently drops or
        # misaligns rows when the column dtype is bool.
        contingency = contingency.reindex(columns=[True, False], fill_value=0)

        if contingency.shape[0] < 2:
            return []

        # Chi-square is unreliable when any expected cell count is below
        # min_expected_freq; in that case skip the global gate and rely on
        # the per-group one-vs-rest Fisher exact tests below.
        expected = expected_freq(contingency.to_numpy())
        fisher_fallback = bool((expected < config.min_expected_freq).any())
        p_global = None
        if not fisher_fallback:
            _, p_global, _, _ = chi2_contingency(contingency)
            if p_global >= config.alpha:
                return []

        overall_event_count = contingency[True].sum()
        overall_nonevent_count = contingency[False].sum()

        tests = []
        for group in contingency.index:
            group_event = contingency.loc[group, True]
            group_nonevent = contingency.loc[group, False]
            table = [
                [group_event, group_nonevent],
                [
                    overall_event_count - group_event,
                    overall_nonevent_count - group_nonevent,
                ],
            ]
            odds_ratio, p_value = fisher_exact(table, alternative="greater")
            tests.append(
                {
                    "group": group,
                    "p_value": float(p_value),
                    "odds_ratio": float(odds_ratio),
                }
            )

        adjusted_p_values = holm_bonferroni_adjust([t["p_value"] for t in tests])

        shared_metrics = {
            "num_hypotheses": float(len(tests)),
            "sample_size": float(len(df)),
            "fisher_fallback": 1.0 if fisher_fallback else 0.0,
        }
        if p_global is not None:
            shared_metrics["p_value_global"] = float(p_global)

        eligible = [
            GroupAssociation(
                group=test["group"],
                p_value=test["p_value"],
                p_value_adjusted=adjusted,
                odds_ratio=test["odds_ratio"],
                metrics=dict(shared_metrics),
            )
            for test, adjusted in zip(tests, adjusted_p_values, strict=True)
            if test["odds_ratio"] > 1 and adjusted < config.alpha
        ]
        # Stable sort keeps contingency-index order on ties, matching the old
        # min()-based single-winner selection exactly.
        eligible.sort(key=lambda item: item.p_value_adjusted)
        return eligible
```

Update `analysis_core/__init__.py` to:

```python
from analysis_core.association import (
    AssociationConfig,
    ChiSquareFisherRanker,
    GroupAssociation,
)
from analysis_core.stats import holm_bonferroni_adjust
from analysis_core.validation import require_columns

__all__ = [
    "AssociationConfig",
    "ChiSquareFisherRanker",
    "GroupAssociation",
    "holm_bonferroni_adjust",
    "require_columns",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m pytest tests/analysis_core/test_association.py -v`
Expected: all PASS.

- [ ] **Step 5: Rewire the adapter — replace `nce_analysis/root_cause/statistical.py` entirely with:**

```python
import pandas as pd

from analysis_core import AssociationConfig, ChiSquareFisherRanker

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import (
    RootCauseCandidate,
    RootCauseStrategy,
    build_suspect_key,
    split_suspect_key,
)


class StatisticalStrategy(RootCauseStrategy):
    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        working = group_df.copy()
        working["suspect_key"] = build_suspect_key(working, config)

        ranked = ChiSquareFisherRanker(
            AssociationConfig(
                group_col="suspect_key",
                event_col="is_anomaly",
                alpha=config.alpha,
            )
        ).rank(working)
        if not ranked:
            return []

        best = ranked[0]
        tool_id, chamber_id = split_suspect_key(best.group, config)
        metrics = {
            "p_value_combo": best.p_value,
            "p_value_combo_adjusted": best.p_value_adjusted,
            "num_hypotheses": best.metrics["num_hypotheses"],
            "multiple_testing_method": 1.0,
            "odds_ratio": best.odds_ratio,
            "sample_size": best.metrics["sample_size"],
            "fisher_fallback": best.metrics["fisher_fallback"],
        }
        if "p_value_global" in best.metrics:
            metrics["p_value_global"] = best.metrics["p_value_global"]
        return [
            RootCauseCandidate(
                suspect_tool_id=tool_id,
                suspect_chamber_id=chamber_id,
                confidence_score=(1 - best.p_value) * 100,
                metrics=metrics,
            )
        ]
```

Note: `_holm_bonferroni_adjust` is deleted from this file — it now lives in `analysis_core/stats.py`. If any test imports it from the old location, that would be a behavior question to raise, not silently fix (existing tests must not be modified).

- [ ] **Step 6: Run the full suite**

Run: `python3.12 -m pytest tests/ -v`
Expected: all PASS — in particular `tests/test_statistical.py` and `tests/test_pipeline_e2e.py`, unmodified.

- [ ] **Step 7: Commit**

```bash
git add analysis_core/ tests/analysis_core/test_association.py nce_analysis/root_cause/statistical.py
git commit -m "refactor: extract chi-square/Fisher association ranking into analysis_core

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `uplift.py` + rewire `MLStrategy`

**Files:**
- Create: `analysis_core/uplift.py`
- Create: `tests/analysis_core/test_uplift.py`
- Modify: `analysis_core/__init__.py`
- Modify: `nce_analysis/root_cause/ml.py` (full rewrite shown below)

**Interfaces:**
- Consumes: `require_columns` (Task 1).
- Produces: `UpliftConfig(group_col, event_col)`, `GroupUplift(group, uplift, normalized_share, metrics)`, `RiskUpliftRanker(config).rank(df) -> list[GroupUplift]` — positive-uplift groups only, sorted by `uplift` descending.

- [ ] **Step 1: Write the failing tests**

`tests/analysis_core/test_uplift.py`:

```python
import pandas as pd
import pytest

from analysis_core.uplift import GroupUplift, RiskUpliftRanker, UpliftConfig


def make_df(rows: list[tuple[str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["group", "is_event"])


def test_ranks_positive_uplift_groups_descending():
    rows = (
        [("worst", True)] * 8 + [("worst", False)] * 2
        + [("mid", True)] * 5 + [("mid", False)] * 5
        + [("clean", False)] * 10
    )
    ranked = RiskUpliftRanker(UpliftConfig()).rank(make_df(rows))
    assert [item.group for item in ranked] == ["worst", "mid"]
    assert ranked[0].uplift > ranked[1].uplift > 0
    assert abs(sum(item.normalized_share for item in ranked) - 1.0) < 1e-9
    assert ranked[0].metrics["group_event_rate"] == 0.8
    assert ranked[0].metrics["group_size"] == 10.0
    assert ranked[0].metrics["sample_size"] == 30.0


def test_single_group_returns_empty():
    rows = [("only", True), ("only", False)]
    assert RiskUpliftRanker(UpliftConfig()).rank(make_df(rows)) == []


def test_all_events_returns_empty():
    rows = [("a", True)] * 3 + [("b", True)] * 3
    assert RiskUpliftRanker(UpliftConfig()).rank(make_df(rows)) == []


def test_no_events_returns_empty():
    rows = [("a", False)] * 3 + [("b", False)] * 3
    assert RiskUpliftRanker(UpliftConfig()).rank(make_df(rows)) == []


def test_custom_column_names():
    df = pd.DataFrame(
        {
            "chamber": ["x"] * 5 + ["y"] * 5,
            "fail": [True] * 4 + [False] * 1 + [False] * 5,
        }
    )
    config = UpliftConfig(group_col="chamber", event_col="fail")
    ranked = RiskUpliftRanker(config).rank(df)
    assert ranked and ranked[0].group == "x"


def test_missing_column_raises():
    with pytest.raises(ValueError, match="is_event"):
        RiskUpliftRanker(UpliftConfig()).rank(pd.DataFrame({"group": ["a"]}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m pytest tests/analysis_core/test_uplift.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'analysis_core.uplift'`.

- [ ] **Step 3: Implement `analysis_core/uplift.py`**

Algorithm moved from `MLStrategy.analyze`:

```python
from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, ConfigDict

from analysis_core.validation import require_columns


class UpliftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_col: str = "group"
    event_col: str = "is_event"


@dataclass
class GroupUplift:
    group: str
    uplift: float
    normalized_share: float
    metrics: dict[str, float] = field(default_factory=dict)


class RiskUpliftRanker:
    """Score each group by its event-rate uplift over the overall event rate.
    Returns only positive-uplift groups, highest uplift first; empty list
    means no group is elevated (or the labels are degenerate)."""

    def __init__(self, config: UpliftConfig):
        self.config = config

    def rank(self, df: pd.DataFrame) -> list[GroupUplift]:
        config = self.config
        require_columns(df, [config.group_col, config.event_col])

        if df[config.group_col].nunique() < 2:
            return []

        labels = df[config.event_col].astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            return []

        overall_rate = float(labels.mean())
        grouped = (
            df.assign(_label=labels)
            .groupby(config.group_col)["_label"]
            .agg(["mean", "sum", "count"])
        )
        grouped["uplift"] = grouped["mean"] - overall_rate
        positive = grouped[grouped["uplift"] > 0]
        if positive.empty:
            return []

        positive_total = float(positive["uplift"].sum())
        results = [
            GroupUplift(
                group=group,
                uplift=float(row["uplift"]),
                normalized_share=(
                    float(row["uplift"]) / positive_total if positive_total > 0 else 0.0
                ),
                metrics={
                    "group_event_rate": float(row["mean"]),
                    "overall_event_rate": overall_rate,
                    "group_event_count": float(row["sum"]),
                    "group_size": float(row["count"]),
                    "sample_size": float(len(df)),
                },
            )
            for group, row in positive.iterrows()
        ]
        # Stable sort keeps groupby-index order on ties, matching the old
        # idxmax()-based single-winner selection exactly.
        results.sort(key=lambda item: item.uplift, reverse=True)
        return results
```

Update `analysis_core/__init__.py` — add to the imports and `__all__`:

```python
from analysis_core.uplift import GroupUplift, RiskUpliftRanker, UpliftConfig
```

and extend `__all__` with `"GroupUplift", "RiskUpliftRanker", "UpliftConfig"` (keep the list alphabetically sorted).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m pytest tests/analysis_core/test_uplift.py -v`
Expected: all PASS.

- [ ] **Step 5: Rewire the adapter — replace `nce_analysis/root_cause/ml.py` entirely with:**

```python
import pandas as pd

from analysis_core import RiskUpliftRanker, UpliftConfig

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import (
    RootCauseCandidate,
    RootCauseStrategy,
    build_suspect_key,
    split_suspect_key,
)


class MLStrategy(RootCauseStrategy):
    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        working = group_df.copy()
        working["suspect_key"] = build_suspect_key(working, config)

        ranked = RiskUpliftRanker(
            UpliftConfig(group_col="suspect_key", event_col="is_anomaly")
        ).rank(working)
        if not ranked:
            return []

        best = ranked[0]
        tool_id, chamber_id = split_suspect_key(best.group, config)
        return [
            RootCauseCandidate(
                suspect_tool_id=tool_id,
                suspect_chamber_id=chamber_id,
                confidence_score=float(best.normalized_share * 100.0),
                metrics={
                    "risk_uplift": best.uplift,
                    "suspect_anomaly_rate": best.metrics["group_event_rate"],
                    "overall_anomaly_rate": best.metrics["overall_event_rate"],
                    "suspect_anomaly_count": best.metrics["group_event_count"],
                    "suspect_sample_size": best.metrics["group_size"],
                    "sample_size": best.metrics["sample_size"],
                    "ml_scoring_method": 1.0,
                },
            )
        ]
```

- [ ] **Step 6: Run the full suite**

Run: `python3.12 -m pytest tests/ -v`
Expected: all PASS — in particular `tests/test_ml.py` and `tests/test_both.py`, unmodified.

- [ ] **Step 7: Commit**

```bash
git add analysis_core/ tests/analysis_core/test_uplift.py nce_analysis/root_cause/ml.py
git commit -m "refactor: extract risk-uplift ranking into analysis_core

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `trend.py` + rewire both drift strategies

**Files:**
- Create: `analysis_core/trend.py`
- Create: `tests/analysis_core/test_trend.py`
- Modify: `analysis_core/__init__.py`
- Modify: `nce_analysis/drift/base.py` (add kind→label mapping)
- Modify: `nce_analysis/drift/regression_cusum.py` (full rewrite shown below)
- Modify: `nce_analysis/drift/correlation.py` (full rewrite shown below)

**Interfaces:**
- Consumes: `require_columns` (Task 1).
- Produces: `TrendConfig(time_col, value_col, alpha, cusum_sigma)`, `TrendResult(kind: Literal["change_point", "increasing_trend", "none"], metrics)`, `RegressionCusumDetector(config).detect(df) -> TrendResult`, `CorrelationTrendDetector(config).detect(df) -> TrendResult`; plus `KIND_TO_ROOT_CAUSE` dict in `nce_analysis.drift.base`.

- [ ] **Step 1: Write the failing tests**

`tests/analysis_core/test_trend.py`:

```python
import numpy as np
import pandas as pd

from analysis_core.trend import (
    CorrelationTrendDetector,
    RegressionCusumDetector,
    TrendConfig,
)


def make_series(values) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=len(values), freq="h")
    return pd.DataFrame({"timestamp": times, "value": list(values)})


def test_noisy_increase_is_increasing_trend():
    rng = np.random.default_rng(42)
    values = np.arange(30) * 2.0 + rng.normal(0, 0.5, 30)
    result = RegressionCusumDetector(TrendConfig()).detect(make_series(values))
    assert result.kind == "increasing_trend"
    assert result.metrics["slope"] > 0
    assert result.metrics["slope_p_value"] < 0.05
    assert result.metrics["cusum_threshold_method"] == 1.0


def test_step_jump_is_change_point():
    rng = np.random.default_rng(7)
    values = np.concatenate(
        [rng.normal(10, 0.3, 15), rng.normal(20, 0.3, 15)]
    )
    result = RegressionCusumDetector(TrendConfig()).detect(make_series(values))
    assert result.kind == "change_point"
    assert result.metrics["cusum_range"] > result.metrics["cusum_threshold"]


def test_flat_noise_is_none():
    rng = np.random.default_rng(3)
    values = rng.normal(10, 0.5, 20)
    result = RegressionCusumDetector(TrendConfig()).detect(make_series(values))
    assert result.kind == "none"


def test_constant_timestamps_report_insufficient_time_variation():
    df = pd.DataFrame(
        {"timestamp": ["2026-01-01"] * 5, "value": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    result = RegressionCusumDetector(TrendConfig()).detect(df)
    assert result.kind == "none"
    assert result.metrics == {"insufficient_time_variation": 1.0}


def test_correlation_detector_flags_increasing_trend():
    values = [float(i) for i in range(12)]
    result = CorrelationTrendDetector(TrendConfig()).detect(make_series(values))
    assert result.kind == "increasing_trend"
    assert result.metrics["correlation"] > 0


def test_correlation_constant_values_report_insufficient_value_variation():
    result = CorrelationTrendDetector(TrendConfig()).detect(make_series([5.0] * 6))
    assert result.kind == "none"
    assert result.metrics == {"insufficient_value_variation": 1.0}


def test_custom_column_names():
    df = pd.DataFrame(
        {
            "ran_at": pd.date_range("2026-01-01", periods=12, freq="h"),
            "reading": [float(i) for i in range(12)],
        }
    )
    config = TrendConfig(time_col="ran_at", value_col="reading")
    result = CorrelationTrendDetector(config).detect(df)
    assert result.kind == "increasing_trend"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m pytest tests/analysis_core/test_trend.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'analysis_core.trend'`.

- [ ] **Step 3: Implement `analysis_core/trend.py`**

Algorithms moved from `RegressionCusum.classify` / `Correlation.classify`; the only functional change is `cusum_sigma` replacing the hard-coded `3.0` (default keeps the same value):

```python
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import stats

from analysis_core.validation import require_columns


class TrendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_col: str = "timestamp"
    value_col: str = "value"
    alpha: float = 0.05
    cusum_sigma: float = 3.0


@dataclass
class TrendResult:
    kind: Literal["change_point", "increasing_trend", "none"]
    metrics: dict[str, float] = field(default_factory=dict)


def _elapsed_hours_and_values(
    df: pd.DataFrame, config: TrendConfig
) -> tuple[np.ndarray, np.ndarray]:
    require_columns(df, [config.time_col, config.value_col])
    ordered = df.sort_values(config.time_col).reset_index(drop=True)
    timestamps = pd.to_datetime(ordered[config.time_col])
    elapsed_hours = (
        (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
    ).to_numpy()
    values = ordered[config.value_col].to_numpy()
    return elapsed_hours, values


class RegressionCusumDetector:
    """Linear regression for a monotone trend plus a CUSUM control-chart pass
    for abrupt change points. change_point outranks increasing_trend."""

    def __init__(self, config: TrendConfig):
        self.config = config

    def detect(self, df: pd.DataFrame) -> TrendResult:
        elapsed_hours, values = _elapsed_hours_and_values(df, self.config)

        if np.unique(elapsed_hours).size < 2:
            return TrendResult("none", {"insufficient_time_variation": 1.0})

        slope, intercept, _, p_value, _ = stats.linregress(elapsed_hours, values)
        fitted = intercept + slope * elapsed_hours
        residuals = values - fitted

        cusum = np.cumsum(residuals - residuals.mean())
        cusum_range = float(cusum.max() - cusum.min())
        residual_std = float(residuals.std())
        n = len(residuals)
        # Threshold scales with sample size so short, noisy series aren't
        # flagged as sudden shifts when they're really just noise.
        cusum_threshold = (
            self.config.cusum_sigma * residual_std * np.sqrt(n)
            if residual_std > 0 and n > 0
            else 0.0
        )
        change_point_detected = cusum_threshold > 0 and cusum_range > cusum_threshold

        metrics = {
            "slope": float(slope),
            "slope_p_value": float(p_value),
            "cusum_range": cusum_range,
            "cusum_threshold": cusum_threshold,
            "cusum_threshold_method": 1.0,
        }

        if change_point_detected:
            return TrendResult("change_point", metrics)
        if p_value < self.config.alpha and slope > 0:
            return TrendResult("increasing_trend", metrics)
        return TrendResult("none", metrics)


class CorrelationTrendDetector:
    """Pearson correlation of value against elapsed time."""

    def __init__(self, config: TrendConfig):
        self.config = config

    def detect(self, df: pd.DataFrame) -> TrendResult:
        elapsed_hours, values = _elapsed_hours_and_values(df, self.config)

        if np.unique(elapsed_hours).size < 2:
            return TrendResult("none", {"insufficient_time_variation": 1.0})
        if np.unique(values).size < 2:
            return TrendResult("none", {"insufficient_value_variation": 1.0})

        correlation, p_value = stats.pearsonr(elapsed_hours, values)
        metrics = {"correlation": float(correlation), "p_value": float(p_value)}

        if p_value < self.config.alpha and correlation > 0:
            return TrendResult("increasing_trend", metrics)
        return TrendResult("none", metrics)
```

Update `analysis_core/__init__.py` — add:

```python
from analysis_core.trend import (
    CorrelationTrendDetector,
    RegressionCusumDetector,
    TrendConfig,
    TrendResult,
)
```

and extend `__all__` with `"CorrelationTrendDetector", "RegressionCusumDetector", "TrendConfig", "TrendResult"` (keep alphabetical order).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m pytest tests/analysis_core/test_trend.py -v`
Expected: all PASS. If `test_noisy_increase_is_increasing_trend` or `test_step_jump_is_change_point` fails on its `kind` assertion, the fixture noise level is at fault (the seeds above were chosen to be safely inside each regime) — adjust the fixture's noise scale, never the detector.

- [ ] **Step 5: Rewire the adapters**

Append to `nce_analysis/drift/base.py` (after the existing `DriftStrategy` class):

```python
KIND_TO_ROOT_CAUSE = {
    "change_point": "CHAMBER_SUDDEN_SHIFT",
    "increasing_trend": "CHAMBER_DRIFT",
    "none": "SPECIFIC_CHAMBER_DEFECT",
}
```

Replace `nce_analysis/drift/regression_cusum.py` entirely with:

```python
import pandas as pd

from analysis_core import RegressionCusumDetector, TrendConfig

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.base import KIND_TO_ROOT_CAUSE, DriftStrategy


class RegressionCusum(DriftStrategy):
    def classify(
        self, series_df: pd.DataFrame, config: AnalysisConfig
    ) -> tuple[str, dict[str, float]]:
        result = RegressionCusumDetector(
            TrendConfig(
                time_col="Pre_Execute_Time",
                value_col="NCE_Value",
                alpha=config.alpha,
            )
        ).detect(series_df)
        return KIND_TO_ROOT_CAUSE[result.kind], result.metrics
```

Replace `nce_analysis/drift/correlation.py` entirely with:

```python
import pandas as pd

from analysis_core import CorrelationTrendDetector, TrendConfig

from nce_analysis.config import AnalysisConfig
from nce_analysis.drift.base import KIND_TO_ROOT_CAUSE, DriftStrategy


class Correlation(DriftStrategy):
    def classify(
        self, series_df: pd.DataFrame, config: AnalysisConfig
    ) -> tuple[str, dict[str, float]]:
        result = CorrelationTrendDetector(
            TrendConfig(
                time_col="Pre_Execute_Time",
                value_col="NCE_Value",
                alpha=config.alpha,
            )
        ).detect(series_df)
        return KIND_TO_ROOT_CAUSE[result.kind], result.metrics
```

- [ ] **Step 6: Run the full suite**

Run: `python3.12 -m pytest tests/ -v`
Expected: all PASS — in particular `tests/test_drift_base.py`, `tests/test_regression_cusum.py`, `tests/test_correlation.py`, unmodified.

- [ ] **Step 7: Commit**

```bash
git add analysis_core/ tests/analysis_core/test_trend.py nce_analysis/drift/
git commit -m "refactor: extract trend/change-point detection into analysis_core

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `ratio_hotspot.py` + rewire `RatioThreshold`

**Files:**
- Create: `analysis_core/ratio_hotspot.py`
- Create: `tests/analysis_core/test_ratio_hotspot.py`
- Modify: `analysis_core/__init__.py`
- Modify: `nce_analysis/hotspot/ratio_threshold.py` (full rewrite shown below)

**Interfaces:**
- Consumes: `require_columns` (Task 1).
- Produces: `RatioHotspotConfig(group_cols, event_col, ratio_threshold, min_count)`, `RatioHotspotDetector(config).detect(df) -> pd.DataFrame` with columns `[*group_cols, event_count, total_count, event_ratio, insufficient_sample]`, filtered to `event_ratio >= ratio_threshold`, sorted by `(event_ratio, event_count)` descending.

- [ ] **Step 1: Write the failing tests**

`tests/analysis_core/test_ratio_hotspot.py`:

```python
import pandas as pd
import pytest

from analysis_core.ratio_hotspot import RatioHotspotConfig, RatioHotspotDetector


def test_groups_filtered_by_ratio_and_sorted_descending():
    df = pd.DataFrame(
        {
            "x": [0] * 10 + [1] * 10 + [2] * 10,
            "y": [0] * 30,
            "is_event": (
                [True] * 8 + [False] * 2      # (0,0): ratio 0.8
                + [True] * 1 + [False] * 9    # (1,0): ratio 0.1
                + [False] * 10                # (2,0): ratio 0.0 -> filtered
            ),
        }
    )
    config = RatioHotspotConfig(
        group_cols=["x", "y"], ratio_threshold=0.05, min_count=5
    )
    detected = RatioHotspotDetector(config).detect(df)
    assert list(detected.columns) == [
        "x", "y", "event_count", "total_count", "event_ratio", "insufficient_sample",
    ]
    assert list(detected["event_ratio"]) == [0.8, 0.1]
    assert list(detected["event_count"]) == [8, 1]
    assert not detected["insufficient_sample"].any()


def test_small_groups_kept_but_flagged_insufficient():
    df = pd.DataFrame(
        {
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "is_event": [True, True, False],
        }
    )
    config = RatioHotspotConfig(
        group_cols=["x", "y"], ratio_threshold=0.05, min_count=5
    )
    detected = RatioHotspotDetector(config).detect(df)
    assert len(detected) == 1
    assert bool(detected.loc[0, "insufficient_sample"]) is True


def test_missing_column_raises():
    config = RatioHotspotConfig(group_cols=["x"], event_col="is_event")
    with pytest.raises(ValueError, match="is_event"):
        RatioHotspotDetector(config).detect(pd.DataFrame({"x": [1]}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m pytest tests/analysis_core/test_ratio_hotspot.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'analysis_core.ratio_hotspot'`.

- [ ] **Step 3: Implement `analysis_core/ratio_hotspot.py`**

Algorithm moved from `RatioThreshold.detect` (minus the domain-specific dedup and anomaly derivation, which stay in the adapter):

```python
import pandas as pd
from pydantic import BaseModel, ConfigDict

from analysis_core.validation import require_columns


class RatioHotspotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_cols: list[str]
    event_col: str = "is_event"
    ratio_threshold: float = 0.05
    min_count: int = 5


class RatioHotspotDetector:
    """Group rows by group_cols and keep groups whose event ratio meets
    ratio_threshold. Groups below min_count rows are kept but flagged
    insufficient_sample=True so callers can report them without scoring."""

    def __init__(self, config: RatioHotspotConfig):
        self.config = config

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        config = self.config
        require_columns(df, [*config.group_cols, config.event_col])

        grouped = (
            df.groupby(config.group_cols)
            .agg(
                event_count=(config.event_col, "sum"),
                total_count=(config.event_col, "count"),
            )
            .reset_index()
        )
        grouped["event_ratio"] = grouped["event_count"] / grouped["total_count"]

        grouped = grouped[grouped["event_ratio"] >= config.ratio_threshold]
        grouped["insufficient_sample"] = grouped["total_count"] < config.min_count

        return grouped.sort_values(
            ["event_ratio", "event_count"], ascending=False
        ).reset_index(drop=True)
```

Update `analysis_core/__init__.py` — add:

```python
from analysis_core.ratio_hotspot import RatioHotspotConfig, RatioHotspotDetector
```

and extend `__all__` with `"RatioHotspotConfig", "RatioHotspotDetector"` (keep alphabetical order).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m pytest tests/analysis_core/test_ratio_hotspot.py -v`
Expected: all PASS.

- [ ] **Step 5: Rewire the adapter — replace `nce_analysis/hotspot/ratio_threshold.py` entirely with:**

```python
import pandas as pd

from analysis_core import RatioHotspotConfig, RatioHotspotDetector

from nce_analysis.config import AnalysisConfig
from nce_analysis.hotspot.base import HotspotStrategy

LITHO_POINT_KEY_COLUMNS = ["WaferID", "X_Posi", "Y_Posi"]


class RatioThreshold(HotspotStrategy):
    def detect(self, long_df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
        litho_points = long_df.drop_duplicates(subset=LITHO_POINT_KEY_COLUMNS).copy()
        litho_points["is_anomaly"] = litho_points["NCE_Value"] > config.spec_threshold

        detected = RatioHotspotDetector(
            RatioHotspotConfig(
                group_cols=["X_Posi", "Y_Posi"],
                event_col="is_anomaly",
                ratio_threshold=config.hotspot_ratio_threshold,
                min_count=config.min_wafer_count,
            )
        ).detect(litho_points)

        return detected.rename(
            columns={
                "event_count": "anomalous_wafer_count",
                "total_count": "total_wafer_count",
                "event_ratio": "anomaly_ratio",
            }
        )
```

- [ ] **Step 6: Run the full suite**

Run: `python3.12 -m pytest tests/ -v`
Expected: all PASS — in particular `tests/test_hotspot_base.py`, `tests/test_ratio_threshold.py`, `tests/test_pipeline_e2e.py`, unmodified.

- [ ] **Step 7: Commit**

```bash
git add analysis_core/ tests/analysis_core/test_ratio_hotspot.py nce_analysis/hotspot/ratio_threshold.py
git commit -m "refactor: extract ratio-threshold hotspot detection into analysis_core

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `dominance.py` + rewire `MajorityRule`

**Files:**
- Create: `analysis_core/dominance.py`
- Create: `tests/analysis_core/test_dominance.py`
- Modify: `analysis_core/__init__.py`
- Modify: `nce_analysis/noise_filter/majority_rule.py` (full rewrite shown below)

**Interfaces:**
- Consumes: nothing new (pure pandas).
- Produces: `DominanceConfig(threshold)`, `Dominance(category: tuple, share: float, count: int)`, `find_dominant(keys: pd.Series, config) -> Dominance | None` — `None` when no category's share is **strictly greater** than `threshold` or the series is empty; scalar categories are returned as 1-tuples.

- [ ] **Step 1: Write the failing tests**

`tests/analysis_core/test_dominance.py`:

```python
import pandas as pd

from analysis_core.dominance import Dominance, DominanceConfig, find_dominant


def test_majority_category_found():
    keys = pd.Series(["a"] * 7 + ["b"] * 3)
    assert find_dominant(keys, DominanceConfig(threshold=0.5)) == Dominance(
        category=("a",), share=0.7, count=7
    )


def test_share_equal_to_threshold_is_not_dominant():
    keys = pd.Series(["a"] * 5 + ["b"] * 5)
    assert find_dominant(keys, DominanceConfig(threshold=0.5)) is None


def test_composite_keys_returned_as_is():
    keys = pd.Series([("t1", "c1")] * 3 + [("t2", "c2")] * 1)
    dominance = find_dominant(keys, DominanceConfig(threshold=0.5))
    assert dominance is not None
    assert dominance.category == ("t1", "c1")
    assert dominance.count == 3


def test_empty_series_returns_none():
    keys = pd.Series([], dtype=object)
    assert find_dominant(keys, DominanceConfig()) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m pytest tests/analysis_core/test_dominance.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'analysis_core.dominance'`.

- [ ] **Step 3: Implement `analysis_core/dominance.py`**

```python
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ConfigDict


class DominanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: float = 0.5


@dataclass
class Dominance:
    category: tuple
    share: float
    count: int


def find_dominant(keys: pd.Series, config: DominanceConfig) -> Dominance | None:
    """Return the most frequent category if its share of `keys` is strictly
    greater than config.threshold, else None. Scalar categories are wrapped
    as 1-tuples so composite keys (already tuples) and scalars have a
    uniform return shape."""
    if keys.empty:
        return None
    counts = keys.value_counts()
    top_category, top_count = counts.index[0], int(counts.iloc[0])
    share = top_count / len(keys)
    if share <= config.threshold:
        return None
    category = top_category if isinstance(top_category, tuple) else (top_category,)
    return Dominance(category=category, share=share, count=top_count)
```

Update `analysis_core/__init__.py` — add:

```python
from analysis_core.dominance import Dominance, DominanceConfig, find_dominant
```

and extend `__all__` with `"Dominance", "DominanceConfig", "find_dominant"` (keep alphabetical order).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m pytest tests/analysis_core/test_dominance.py -v`
Expected: all PASS.

- [ ] **Step 5: Rewire the adapter — replace `nce_analysis/noise_filter/majority_rule.py` entirely with:**

The two-tier check order (Tool+Chuck combo first, then Tool), stage-diversity
contamination rule, and all LITHO_* labels are domain logic and stay here;
only the "is one category dominant" kernel is delegated.

```python
import pandas as pd

from analysis_core import DominanceConfig, find_dominant

from nce_analysis.config import AnalysisConfig
from nce_analysis.noise_filter.base import NoiseFilterResult, NoiseFilterStrategy
from nce_analysis.schema import RootCauseDetail


class MajorityRule(NoiseFilterStrategy):
    def filter(
        self, long_df: pd.DataFrame, hotspots: pd.DataFrame, config: AnalysisConfig
    ) -> NoiseFilterResult:
        litho_points = long_df.drop_duplicates(subset=["WaferID", "X_Posi", "Y_Posi"]).copy()
        litho_points["is_anomaly"] = litho_points["NCE_Value"] > config.spec_threshold

        dominance_config = DominanceConfig(threshold=config.noise_filter_majority_threshold)
        surviving_rows = []
        self_issues: list[RootCauseDetail] = []

        for _, hotspot in hotspots.iterrows():
            coord_mask = (
                (litho_points["X_Posi"] == hotspot["X_Posi"])
                & (litho_points["Y_Posi"] == hotspot["Y_Posi"])
                & litho_points["is_anomaly"]
            )
            anomalous = litho_points[coord_mask]
            if len(anomalous) == 0:
                surviving_rows.append(hotspot)
                continue

            chuck_dominance = find_dominant(
                pd.Series(list(zip(anomalous["ToolID"], anomalous["ChuckID"]))),
                dominance_config,
            )
            if chuck_dominance is not None:
                owning_tool, top_chuck = chuck_dominance.category
                chuck_rows = anomalous[
                    (anomalous["ToolID"] == owning_tool) & (anomalous["ChuckID"] == top_chuck)
                ]
                stage_diversity = chuck_rows["StageID"].nunique()
                root_cause_type = (
                    "LITHO_CHUCK_CONTAMINATION" if stage_diversity > 1 else "LITHO_CHUCK_ISSUE"
                )
                self_issues.append(
                    RootCauseDetail(
                        Suspect_Pre_ToolID=owning_tool,
                        Suspect_Pre_ChamberID=top_chuck,
                        Suspect_Pre_StepID="N/A",
                        Confidence_Score=chuck_dominance.share * 100,
                        Root_Cause_Type=root_cause_type,
                        Affected_Coordinates=[(hotspot["X_Posi"], hotspot["Y_Posi"])],
                        Metrics={
                            "chuck_share": chuck_dominance.share,
                            "stage_diversity": float(stage_diversity),
                        },
                    )
                )
                continue

            tool_dominance = find_dominant(anomalous["ToolID"], dominance_config)
            if tool_dominance is not None:
                (top_tool,) = tool_dominance.category
                self_issues.append(
                    RootCauseDetail(
                        Suspect_Pre_ToolID=top_tool,
                        Suspect_Pre_ChamberID="N/A",
                        Suspect_Pre_StepID="N/A",
                        Confidence_Score=tool_dominance.share * 100,
                        Root_Cause_Type="LITHO_TOOL_ISSUE",
                        Affected_Coordinates=[(hotspot["X_Posi"], hotspot["Y_Posi"])],
                        Metrics={"tool_share": tool_dominance.share},
                    )
                )
                continue

            surviving_rows.append(hotspot)

        surviving_df = (
            pd.DataFrame(surviving_rows) if surviving_rows else hotspots.iloc[0:0].copy()
        )
        return NoiseFilterResult(surviving_hotspots=surviving_df, litho_self_issues=self_issues)
```

- [ ] **Step 6: Run the full suite**

Run: `python3.12 -m pytest tests/ -v`
Expected: all PASS — in particular `tests/test_majority_rule.py`, `tests/test_noise_filter_base.py`, `tests/test_pipeline_e2e.py`, unmodified.

- [ ] **Step 7: Commit**

```bash
git add analysis_core/ tests/analysis_core/test_dominance.py nce_analysis/noise_filter/majority_rule.py
git commit -m "refactor: extract dominance kernel into analysis_core

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Documentation — CLAUDE.md and README

**Files:**
- Modify: `CLAUDE.md` (Architecture section)
- Modify: `README.md` (Development section)

**Interfaces:**
- Consumes: the final two-layer structure from Tasks 1–6.
- Produces: docs only.

- [ ] **Step 1: Update `CLAUDE.md`**

In the `## Architecture` section, replace the directory tree code block with:

```
analysis_core/           # domain-free algorithm layer (no nce_analysis imports — enforced by test)
├── stats.py             # holm_bonferroni_adjust
├── association.py       # ChiSquareFisherRanker: chi²-gated Fisher one-vs-rest + Holm
├── uplift.py            # RiskUpliftRanker: event-rate uplift ranking
├── trend.py             # RegressionCusumDetector / CorrelationTrendDetector
├── dominance.py         # find_dominant: majority-share category detection
└── ratio_hotspot.py     # RatioHotspotDetector: grouped event-ratio thresholding

nce_analysis/            # domain layer: thin adapters over analysis_core
├── preprocessing/wide_history_reshape.py   # WideHistoryReshape (default; not extracted)
├── hotspot/ratio_threshold.py              # RatioThreshold (default)
├── noise_filter/majority_rule.py           # MajorityRule (default)
├── root_cause/{statistical,ml,both}.py     # StatisticalStrategy / MLStrategy / BothStrategy
├── drift/{regression_cusum,correlation}.py # RegressionCusum (default) / Correlation
├── config.py         # AnalysisConfig — picks the concrete strategy per stage
├── pipeline.py        # run(raw_df, config) -> AnalysisResult — orchestrates all stages
├── schema.py           # pydantic input/output models (RootCauseDetail, AnalysisResult, ...)
└── result.py            # aggregate_results: Details -> deduped/sorted Summary
```

Directly after the tree, add this paragraph:

> Each `nce_analysis` strategy is a thin adapter: it maps domain columns onto
> an `analysis_core` component's neutral column contract (via that
> component's pydantic config), runs it, and converts the neutral result back
> to domain labels (e.g. `TrendResult.kind` → `CHAMBER_DRIFT`) and the
> existing metric keys. `analysis_core` must never import `nce_analysis` or
> use domain vocabulary — `tests/analysis_core/test_dependency_direction.py`
> enforces this. Algorithm changes belong in `analysis_core`; label/metric
> naming and domain thresholds belong in the adapters. Design:
> `docs/superpowers/specs/2026-07-08-analysis-core-extraction-design.md`.

- [ ] **Step 2: Update `README.md`**

In the `## Development` section, after the sentence pointing at `CLAUDE.md`, add:

> Generic, domain-free statistical building blocks (association ranking,
> uplift ranking, trend/change-point detection, dominance, ratio hotspots)
> live in the `analysis_core` package and are reusable outside the NCE
> domain; the `nce_analysis` strategies are thin adapters over them.

- [ ] **Step 3: Run the full suite one last time**

Run: `python3.12 -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: describe analysis_core / nce_analysis two-layer architecture

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
