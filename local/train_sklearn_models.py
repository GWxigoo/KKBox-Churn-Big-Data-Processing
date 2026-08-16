#!/usr/bin/env python3
"""Matched local Logistic Regression comparison for the reduced-scope study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "membership_tenure_days",
    "transaction_count_90",
    "auto_renew_rate_90",
    "cancellation_count_90",
    "days_since_last_transaction",
    "days_until_expiration",
    "no_recent_transaction",
    "active_days_30",
    "total_secs_30",
    "days_since_last_listen",
    "no_recent_listening",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-base", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_parts(folder):
    paths = sorted(Path(folder).glob("*.csv.gz"))
    if not paths:
        paths = sorted(Path(folder).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV part files found below {folder}")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def choose_threshold(labels, probabilities):
    rows = []
    for threshold in np.arange(0.10, 0.901, 0.05):
        predicted = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", zero_division=0
        )
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["f1", "recall", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(best["threshold"]), table


def main():
    args = arguments()
    started = time.perf_counter()
    process = psutil.Process()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    february = read_parts(Path(args.input_base) / "february")
    march = read_parts(Path(args.input_base) / "march")
    if "split_bucket" not in february.columns:
        raise ValueError("February export must contain Spark's split_bucket column")

    development_train = february.loc[february["split_bucket"] < 80].copy()
    validation = february.loc[february["split_bucket"] >= 80].copy()

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=100,
                    penalty=None,
                    solver="lbfgs",
                    random_state=args.seed,
                ),
            ),
        ]
    )

    training_started = time.perf_counter()
    pipeline.fit(
        development_train[FEATURE_COLUMNS], development_train["is_churn"]
    )
    training_seconds = time.perf_counter() - training_started

    validation_started = time.perf_counter()
    validation_probability = pipeline.predict_proba(
        validation[FEATURE_COLUMNS]
    )[:, 1]
    validation_seconds = time.perf_counter() - validation_started
    threshold, threshold_table = choose_threshold(
        validation["is_churn"].to_numpy(), validation_probability
    )

    inference_started = time.perf_counter()
    probability = pipeline.predict_proba(march[FEATURE_COLUMNS])[:, 1]
    inference_seconds = time.perf_counter() - inference_started
    predicted = (probability >= threshold).astype(int)
    labels = march["is_churn"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )

    metrics = {
        "platform": "Local Python/scikit-learn",
        "model": "Logistic Regression - all combined features",
        "training_rows": int(len(development_train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(march)),
        "feature_count": len(FEATURE_COLUMNS),
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(labels, probability)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "log_loss": float(log_loss(labels, probability)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "training_seconds": training_seconds,
        "validation_prediction_seconds": validation_seconds,
        "test_inference_seconds": inference_seconds,
        "end_to_end_seconds": time.perf_counter() - started,
        "rss_mib_at_completion": process.memory_info().rss / (1024**2),
    }

    pd.DataFrame([metrics]).to_csv(output / "sklearn_logistic_metrics.csv", index=False)
    threshold_table.to_csv(output / "sklearn_threshold_search.csv", index=False)
    pd.DataFrame(
        {
            "msno": march["msno"],
            "is_churn": labels,
            "probability": probability,
            "predicted_label": predicted,
        }
    ).to_csv(output / "sklearn_logistic_predictions.csv", index=False)
    (output / "sklearn_run_manifest.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Wrote matched local Logistic Regression results to {output}")


if __name__ == "__main__":
    main()
