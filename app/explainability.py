"""
explainability.py — Layer 3: SHAP-based explainability.

Computes the top-N contributing features for ONE specific prediction
(local explanation), not just global feature importance. This is what lets
the LLM layer say "your period gap and elevated LH/FSH ratio contributed
most to this result" instead of a generic list.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import shap

# Human-readable labels for plain-language explanations (used by llm_service too)
FEATURE_LABELS = {
    "age": "Age",
    "bmi": "Body Mass Index (BMI)",
    "systolic_bp": "Systolic blood pressure",
    "diastolic_bp": "Diastolic blood pressure",
    "pulse_rate": "Pulse rate",
    "period_gap_days": "Gap between periods",
    "stress_score": "Stress / mental load score",
    "fsh": "FSH hormone level",
    "lh": "LH hormone level",
    "lh_fsh_ratio": "LH:FSH ratio",
    "amh": "AMH hormone level",
    "tsh": "Thyroid (TSH) level",
    "prolactin": "Prolactin level",
    "testosterone": "Testosterone level",
    "follicle_count_l": "Left ovary follicle count",
    "follicle_count_r": "Right ovary follicle count",
    "ovary_volume_l": "Left ovary volume",
    "ovary_volume_r": "Right ovary volume",
    "hair_loss": "Hair loss",
    "hair_growth_excessive": "Excess facial/body hair growth",
    "weight_fluctuation": "Recent weight fluctuation",
    "workout_routine": "Exercise routine",
    "pcod_ultrasound_score": "Ultrasound PCOD finding",
    "period_cycle_score": "Cycle regularity",
    "junk_food_score": "Junk food frequency",
    "marital_score": "Marital status",
    "skin_issue_score": "Skin issues (acne/pigmentation)",
}


class ShapExplainer:
    """Wraps a fitted tree model with a cached SHAP TreeExplainer."""

    def __init__(self, model, feature_order: list[str]):
        self.model = model
        self.feature_order = feature_order
        self._explainer = shap.TreeExplainer(model)

    def top_factors(self, row_df: pd.DataFrame, top_n: int = 3) -> list[dict]:
        """
        Returns the top_n features driving THIS prediction toward higher
        (or lower) PCOS risk, as a list of
        {feature, label, shap_value, direction, input_value}.
        """
        shap_values = self._explainer.shap_values(row_df)

        # Handle both binary-classifier list output and single-array output
        if isinstance(shap_values, list):
            sv = np.array(shap_values[1][0]) if len(shap_values) > 1 else np.array(shap_values[0][0])
        else:
            sv = np.array(shap_values[0])
            if sv.ndim > 1:
                sv = sv[:, 1] if sv.shape[1] > 1 else sv[:, 0]

        order = np.argsort(-np.abs(sv))[:top_n]
        results = []
        for idx in order:
            feat = self.feature_order[idx]
            val = float(sv[idx])
            results.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                "shap_value": round(val, 4),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
                "input_value": row_df.iloc[0][feat],
            })
        return results
