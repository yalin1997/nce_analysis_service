import pandas as pd
import shap
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from nce_analysis.config import AnalysisConfig
from nce_analysis.root_cause.base import RootCauseCandidate, RootCauseStrategy


class MLStrategy(RootCauseStrategy):
    def analyze(
        self, group_df: pd.DataFrame, config: AnalysisConfig
    ) -> list[RootCauseCandidate]:
        working = group_df.copy()
        working["combo"] = working["Pre_ToolID"] + "|" + working["Pre_ChamberID"]

        if working["combo"].nunique() < 2:
            return []

        encoder = OneHotEncoder(sparse_output=False)
        features = encoder.fit_transform(working[["combo"]])
        labels = working["is_anomaly"].astype(int).to_numpy()

        if labels.sum() == 0 or labels.sum() == len(labels):
            return []

        model = DecisionTreeClassifier(max_depth=3, random_state=0)
        model.fit(features, labels)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        # Binary classifiers: older shap returns list[class0_array, class1_array];
        # newer shap returns one ndarray with a trailing class axis.
        if isinstance(shap_values, list):
            positive_shap = shap_values[1]
        elif shap_values.ndim == 3:
            positive_shap = shap_values[..., 1]
        else:
            positive_shap = shap_values

        # Per-feature means across the whole dataset cancel out for mutually
        # exclusive one-hot columns (each column's positive and negative pushes
        # balance to zero over a symmetric dataset), so combos are compared by
        # each row's total SHAP push (summed across features) averaged within
        # its own combo group instead.
        row_contribution = positive_shap.sum(axis=1)
        working = working.assign(_row_contribution=row_contribution)
        combo_means = working.groupby("combo")["_row_contribution"].mean()

        best_combo = combo_means.idxmax()
        best_value = combo_means[best_combo]
        if best_value <= 0:
            return []

        tool_id, chamber_id = best_combo.split("|", 1)
        positive_total = combo_means[combo_means > 0].sum()
        confidence_score = (best_value / positive_total) * 100 if positive_total > 0 else 0.0

        return [
            RootCauseCandidate(
                suspect_tool_id=tool_id,
                suspect_chamber_id=chamber_id,
                confidence_score=float(confidence_score),
                metrics={
                    "shap_contribution": float(best_value),
                    "sample_size": float(len(working)),
                },
            )
        ]
