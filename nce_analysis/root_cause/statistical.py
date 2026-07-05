import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats.contingency import expected_freq

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import (
    RootCauseCandidate,
    RootCauseStrategy,
    build_suspect_key,
    split_suspect_key,
)


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


class StatisticalStrategy(RootCauseStrategy):
    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        working = group_df.copy()
        working["suspect_key"] = build_suspect_key(working, config)

        contingency = pd.crosstab(working["suspect_key"], working["is_anomaly"])
        # Use reindex (not `contingency[[True, False]]`) to select/order the
        # True/False columns: pandas treats a literal [True, False] list as a
        # boolean row mask rather than column labels, which silently drops or
        # misaligns rows when the column dtype is bool.
        contingency = contingency.reindex(columns=[True, False], fill_value=0)

        if contingency.shape[0] < 2:
            return []

        # Chi-square is unreliable when any expected cell count is < 5; in
        # that case skip the global gate and rely on the per-combo one-vs-rest
        # Fisher exact tests below (spec §7 fallback behavior).
        expected = expected_freq(contingency.to_numpy())
        fisher_fallback = bool((expected < 5).any())
        p_global = None
        if not fisher_fallback:
            _, p_global, _, _ = chi2_contingency(contingency)
            if p_global >= config.alpha:
                return []

        overall_anomaly_count = contingency[True].sum()
        overall_normal_count = contingency[False].sum()

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

        tool_id, chamber_id = split_suspect_key(best_suspect_key, config)
        confidence_score = (1 - best_p) * 100
        metrics = {
            "p_value_combo": float(best_p),
            "p_value_combo_adjusted": float(best_p_adjusted),
            "num_hypotheses": float(len(test_results)),
            "multiple_testing_method": 1.0,
            "odds_ratio": float(best_odds_ratio),
            "sample_size": float(len(working)),
            "fisher_fallback": 1.0 if fisher_fallback else 0.0,
        }
        if p_global is not None:
            metrics["p_value_global"] = float(p_global)
        return [
            RootCauseCandidate(
                suspect_tool_id=tool_id,
                suspect_chamber_id=chamber_id,
                confidence_score=confidence_score,
                metrics=metrics,
            )
        ]
