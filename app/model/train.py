"""
train.py — trains the PCOS risk classifier (Layer 2).

Usage:
    python app/model/train.py                       # uses data/pcos_dataset.csv
    python app/model/train.py --data path/to/csv     # use a custom dataset

If data/pcos_dataset.csv is missing, this script auto-generates a synthetic
one via data/generate_synthetic_data.py so the pipeline still runs end to
end. Swap in the real Kaggle CSV any time and re-run this script — nothing
else in the app needs to change.

Outputs:
    app/model/pcos_model.pkl   -> {model, medians, feature_order, metrics}
"""

import argparse
import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.preprocessing import dataframe_to_training_matrix, FULL_FEATURE_ORDER  # noqa: E402

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import RandomForestClassifier


def load_or_generate(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        print(f"[train.py] {data_path} not found — generating synthetic dataset instead.")
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from data.generate_synthetic_data import generate
        df = generate()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
    return pd.read_csv(data_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/pcos_dataset.csv")
    parser.add_argument("--out", default="app/model/pcos_model.pkl")
    args = parser.parse_args()

    df = load_or_generate(args.data)
    print(f"[train.py] Loaded dataset with shape {df.shape}")

    X, y, medians = dataframe_to_training_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if HAS_XGB:
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )
        print(f"[train.py] Training XGBoost (scale_pos_weight={scale_pos_weight:.2f})")
    else:
        model = RandomForestClassifier(
            n_estimators=400, max_depth=8, class_weight="balanced", random_state=42
        )
        print("[train.py] xgboost not available — training RandomForest with class_weight='balanced'")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== Evaluation on held-out test set =====")
    print(f"Accuracy : {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall   : {metrics['recall']}")
    print(f"F1 score : {metrics['f1']}")
    print("Confusion matrix:")
    print(cm)
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    artifact = {
        "model": model,
        "medians": medians,
        "feature_order": FULL_FEATURE_ORDER,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "model_type": "xgboost" if HAS_XGB else "random_forest",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\n[train.py] Saved model artifact -> {args.out}")

    with open(os.path.join(os.path.dirname(args.out), "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
