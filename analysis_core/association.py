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
