"""
src/evaluate.py — Load, evaluate, and predict with a trained pipeline
"""

import json
import os
import pickle
from typing import Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PIPELINE_PATH = "models/pipeline.pkl"


def load_model(path: str = PIPELINE_PATH):
    """Load a saved sklearn Pipeline from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No pipeline found at '{path}'. Run main.py first to train and save a model."
        )

    with open(path, "rb") as f:
        pipeline = pickle.load(f)

    print(f"  Pipeline loaded from: {path}")
    return pipeline


def evaluate_model(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pipeline=None,
    model_path: Optional[str] = PIPELINE_PATH,
    save_report: bool = True,
    report_path: str = "models/report.json",
) -> dict:
    """
    Evaluate a fitted pipeline. If no pipeline is passed, load it from model_path.
    """
    if pipeline is None:
        pipeline = load_model(model_path)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print("\n  ── Test Metrics ─────────────────────────")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:<12} {value}")

    print("\n" + classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    if save_report:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Evaluation report saved to: {report_path}")

    return metrics


def predict_new(pipeline, X: pd.DataFrame):
    """
    Predict churn on raw new data using the full fitted pipeline.

    Returns:
        preds: 0/1 predictions.
        probas: churn probabilities.
    """
    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)[:, 1]
    return preds, probas
