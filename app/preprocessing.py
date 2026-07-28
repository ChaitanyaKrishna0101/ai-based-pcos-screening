"""
preprocessing.py — Layer 1: Input validation & preprocessing.

This module is imported by BOTH app/model/train.py and app/main.py so that
the exact same encoding/scaling logic is used at training time and at
inference time (prevents train/serve skew).

Design notes:
- Basic-only users (no doctor consultation) will not have hormone/ultrasound
  fields. Those are treated as missing and imputed with training-set
  medians (stored in the model artifact), never crashing the pipeline.
- Categorical fields are encoded with fixed, explicit mappings (not
  LabelEncoder fit at inference time) so encoding never drifts.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ---- Fixed feature schema -------------------------------------------------

NUMERIC_BASIC = [
    "age", "bmi", "systolic_bp", "diastolic_bp", "pulse_rate",
    "period_gap_days", "stress_score",
]

NUMERIC_HORMONE = [
    "fsh", "lh", "lh_fsh_ratio", "amh", "tsh", "prolactin", "testosterone",
    "follicle_count_l", "follicle_count_r", "ovary_volume_l", "ovary_volume_r",
]

BINARY_YES_NO = [
    "hair_loss", "hair_growth_excessive", "weight_fluctuation", "workout_routine",
]

# pcod_ultrasound has 3 states
ULTRASOUND_MAP = {"No": 0, "Not done": 0.5, "Yes": 1}

PERIOD_CYCLE_MAP = {"Regular": 0, "Irregular": 1}

JUNK_FOOD_MAP = {"Never": 0, "Sometimes": 1, "Often": 2, "Daily": 3}

MARITAL_MAP = {"Unmarried": 0, "Married": 1}

# skin_issues can be multi-select in the UI; collapse to a 0/1 "has_skin_issue"
SKIN_ISSUE_NONE_VALUES = {"none", ""}

FULL_FEATURE_ORDER = (
    NUMERIC_BASIC
    + NUMERIC_HORMONE
    + BINARY_YES_NO
    + ["pcod_ultrasound_score", "period_cycle_score", "junk_food_score",
       "marital_score", "skin_issue_score"]
)


def _yes_no(v) -> int:
    if v is None:
        return 0
    return 1 if str(v).strip().lower() == "yes" else 0


def _skin_issue_score(v) -> int:
    if v is None:
        return 0
    if isinstance(v, (list, tuple, set)):
        vals = [str(x).strip().lower() for x in v]
        return 0 if all(x in SKIN_ISSUE_NONE_VALUES for x in vals) or len(vals) == 0 else 1
    s = str(v).strip().lower()
    return 0 if s in SKIN_ISSUE_NONE_VALUES else 1


def compute_bmi(height_cm: float, weight_kg: float) -> float:
    h_m = height_cm / 100.0
    if h_m <= 0:
        return 0.0
    return round(weight_kg / (h_m ** 2), 2)


def raw_input_to_row(payload: dict, medians: dict | None = None) -> pd.DataFrame:
    """
    Convert a raw JSON payload (as sent by the frontend) into a single-row
    DataFrame with the exact FULL_FEATURE_ORDER columns, imputing missing
    optional (hormone/ultrasound) fields with training medians.

    `medians` is a dict of {feature_name: median_value} learned at training
    time and stored inside the model artifact. If None, zeros are used
    (only acceptable for local smoke-testing, never in production).
    """
    medians = medians or {}
    row = {}

    height = float(payload.get("height_cm", 0) or 0)
    weight = float(payload.get("weight_kg", 0) or 0)
    bmi = payload.get("bmi") or compute_bmi(height, weight)

    row["age"] = float(payload.get("age", 0) or 0)
    row["bmi"] = float(bmi)
    row["systolic_bp"] = float(payload.get("systolic_bp") or medians.get("systolic_bp", 118))
    row["diastolic_bp"] = float(payload.get("diastolic_bp") or medians.get("diastolic_bp", 78))
    row["pulse_rate"] = float(payload.get("pulse_rate") or medians.get("pulse_rate", 78))
    row["period_gap_days"] = float(payload.get("period_gap_days") or medians.get("period_gap_days", 29))
    row["stress_score"] = float(payload.get("stress_score") or medians.get("stress_score", 5))

    for f in NUMERIC_HORMONE:
        val = payload.get(f, None)
        if val in (None, "", "null"):
            row[f] = medians.get(f, 0.0)
        else:
            try:
                row[f] = float(val)
            except (TypeError, ValueError):
                row[f] = medians.get(f, 0.0)

    # derive lh_fsh_ratio if both provided but ratio wasn't
    if payload.get("lh") and payload.get("fsh") and not payload.get("lh_fsh_ratio"):
        try:
            fsh_v = float(payload["fsh"])
            row["lh_fsh_ratio"] = float(payload["lh"]) / fsh_v if fsh_v else medians.get("lh_fsh_ratio", 1.0)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    row["hair_loss"] = _yes_no(payload.get("hair_loss"))
    row["hair_growth_excessive"] = _yes_no(payload.get("hair_growth_excessive"))
    row["weight_fluctuation"] = _yes_no(payload.get("weight_fluctuation"))
    row["workout_routine"] = _yes_no(payload.get("workout_routine"))

    row["pcod_ultrasound_score"] = ULTRASOUND_MAP.get(payload.get("pcod_ultrasound", "Not done"), 0.5)
    row["period_cycle_score"] = PERIOD_CYCLE_MAP.get(payload.get("period_cycle", "Regular"), 0)
    row["junk_food_score"] = JUNK_FOOD_MAP.get(payload.get("junk_food_frequency", "Sometimes"), 1)
    row["marital_score"] = MARITAL_MAP.get(payload.get("marital_status", "Unmarried"), 0)
    row["skin_issue_score"] = _skin_issue_score(payload.get("skin_issues"))

    df = pd.DataFrame([row])[FULL_FEATURE_ORDER]
    return df


def dataframe_to_training_matrix(df: pd.DataFrame):
    """
    Used only by train.py: takes the raw synthetic/Kaggle CSV columns and
    produces (X, y, medians) using the exact same encodings as raw_input_to_row.
    """
    out = pd.DataFrame()
    out["age"] = df["age"].astype(float)
    out["bmi"] = df["bmi"].astype(float)
    out["systolic_bp"] = df["systolic_bp"].astype(float)
    out["diastolic_bp"] = df["diastolic_bp"].astype(float)
    out["pulse_rate"] = df["pulse_rate"].astype(float)
    out["period_gap_days"] = df["period_gap_days"].astype(float)
    out["stress_score"] = df["stress_score"].astype(float)

    for f in NUMERIC_HORMONE:
        out[f] = pd.to_numeric(df[f], errors="coerce")

    out["hair_loss"] = df["hair_loss"].map(_yes_no)
    out["hair_growth_excessive"] = df["hair_growth_excessive"].map(_yes_no)
    out["weight_fluctuation"] = df["weight_fluctuation"].map(_yes_no)
    out["workout_routine"] = df["workout_routine"].map(_yes_no)

    out["pcod_ultrasound_score"] = df["pcod_ultrasound"].map(ULTRASOUND_MAP).fillna(0.5)
    out["period_cycle_score"] = df["period_cycle"].map(PERIOD_CYCLE_MAP).fillna(0)
    out["junk_food_score"] = df["junk_food_frequency"].map(JUNK_FOOD_MAP).fillna(1)
    out["marital_score"] = df["marital_status"].map(MARITAL_MAP).fillna(0)
    out["skin_issue_score"] = df["skin_issues"].map(_skin_issue_score)

    medians = out[NUMERIC_HORMONE].median(numeric_only=True).to_dict()
    medians.update(out[NUMERIC_BASIC].median(numeric_only=True).to_dict())
    out[NUMERIC_HORMONE] = out[NUMERIC_HORMONE].fillna(medians)

    out = out[FULL_FEATURE_ORDER]
    y = df["pcos_diagnosis"].astype(int)
    return out, y, medians
