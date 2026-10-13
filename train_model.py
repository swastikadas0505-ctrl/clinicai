"""ClinicAI - data preparation, model training and evaluation.
Run:  python train_model.py
NOTE: The dataset is SYNTHETIC (generated here). It is NOT real patient data.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

DATA_PATH = "data/appointments.csv"
MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"
FEATURES = ["age", "previous_appointments", "previous_missed",
            "waiting_days", "reminder_sent"]
TARGET = "missed"  # 1 = missed appointment, 0 = attended


def generate_synthetic_data(n=5000, seed=42):
    """Create a reproducible synthetic demonstration dataset."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(40, 18, n), 1, 95).astype(int)
    prev_appts = rng.poisson(4, n)
    prev_missed = rng.binomial(prev_appts, rng.beta(1.5, 8, n))
    waiting_days = np.clip(rng.exponential(12, n), 0, 90).astype(int)
    reminder = rng.binomial(1, 0.6, n)

    miss_rate = prev_missed / np.maximum(prev_appts, 1)
    logit = (-2.0 + 0.04 * waiting_days + 2.5 * miss_rate
             - 0.9 * reminder + 0.012 * (45 - age).clip(min=0))
    prob = 1 / (1 + np.exp(-logit))
    missed = rng.binomial(1, prob)  # random noise keeps it realistic, not perfect

    return pd.DataFrame({
        "age": age, "previous_appointments": prev_appts,
        "previous_missed": prev_missed, "waiting_days": waiting_days,
        "reminder_sent": reminder, TARGET: missed,
    })


def main():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        generate_synthetic_data().to_csv(DATA_PATH, index=False)
        print("Generated SYNTHETIC dataset ->", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Basic cleaning: fill missing numeric values with the median
    df = df.fillna(df.median(numeric_only=True))

    X = df[FEATURES]          # target column is NOT included here (no leakage)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                   class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "test_size": int(len(y_test)),
        "data_type": "synthetic",
    }
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Test-set results (SYNTHETIC data):")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {k}: {metrics[k]:.3f}")
    print("  confusion matrix [[TN, FP], [FN, TP]]:", cm.tolist())
    print("Saved:", MODEL_PATH, "and", METRICS_PATH)


if __name__ == "__main__":
    main()
