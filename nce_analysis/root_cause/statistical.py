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

        best_suspect_key = None
        best_p = 1.0
        best_odds_ratio = 0.0
        for suspect_key in contingency.index:
            suspect_anomaly = contingency.loc[suspect_key, True]
            suspect_normal = contingency.loc[suspect_key, False]
            rest_anomaly = overall_anomaly_count - suspect_anomaly
            rest_normal = overall_normal_count - suspect_normal
            table = [[suspect_anomaly, suspect_normal], [rest_anomaly, rest_normal]]
            odds_ratio, p_value = fisher_exact(table, alternative="greater")
            if p_value < best_p and odds_ratio > 1:
                best_p = p_value
                best_suspect_key = suspect_key
                best_odds_ratio = odds_ratio

        if best_suspect_key is None or best_p >= config.alpha:
            return []

        tool_id, chamber_id = split_suspect_key(best_suspect_key, config)
        confidence_score = (1 - best_p) * 100
        metrics = {
            "p_value_combo": float(best_p),
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
