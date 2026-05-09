"""
src/train.py — Train, tune, compare models, and save the best pipeline
"""

import os
import pickle
from typing import Dict, Tuple, Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .data_prep import build_preprocessor


RANDOM_STATE = 42

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "params": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["liblinear"],
            "model__class_weight": [None, "balanced"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None],
            "model__min_samples_split": [2, 5],
            "model__class_weight": [None, "balanced"],
        },
    },
    "gradient_boosting": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    },
}


def build_pipeline(estimator) -> Pipeline:
    """Create a full sklearn Pipeline: preprocessing + model."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])


def save_model(pipeline: Pipeline, path: str = "models/pipeline.pkl") -> None:
    """Save the fitted best pipeline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n  Best pipeline saved to: {path}")


def train_and_select_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "roc_auc",
    cv: int = 3,
    n_jobs: int = 1,
    save_path: str = "models/pipeline.pkl",
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Train and tune multiple models, select the best one, and save it.

    Args:
        X_train: Raw training features.
        y_train: Training target.
        scoring: Metric used by GridSearchCV to select the best model.
        cv: Number of cross-validation folds.
        n_jobs: Parallel jobs. Use 1 for stable classroom/project environments.
        save_path: Where to save the selected fitted pipeline.

    Returns:
        best_pipeline: The fitted best sklearn Pipeline.
        results_df: Model comparison table sorted by CV score.
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    best_pipeline = None
    best_model_name = None
    best_score = -float("inf")
    results = []

    print("\n  Training and tuning candidate models...")

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n  ── {model_name} ─────────────────────────")

        pipeline = build_pipeline(config["estimator"])
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=config["params"],
            scoring=scoring,
            cv=splitter,
            n_jobs=n_jobs,
            verbose=0,
        )
        search.fit(X_train, y_train)

        model_score = search.best_score_
        print(f"  Best CV {scoring}: {model_score:.4f}")
        print(f"  Best params: {search.best_params_}")

        results.append({
            "model": model_name,
            f"best_cv_{scoring}": round(model_score, 4),
            "best_params": search.best_params_,
        })

        if model_score > best_score:
            best_score = model_score
            best_model_name = model_name
            best_pipeline = search.best_estimator_

    results_df = pd.DataFrame(results).sort_values(
        by=f"best_cv_{scoring}", ascending=False
    ).reset_index(drop=True)

    print("\n  ── Model Comparison ─────────────────────")
    print(results_df[["model", f"best_cv_{scoring}"]].to_string(index=False))
    print(f"\n  Selected model: {best_model_name} | CV {scoring}: {best_score:.4f}")

    save_model(best_pipeline, save_path)
    return best_pipeline, results_df


# Backward-compatible alias for older main.py versions.
def train_model(X_train, y_train, model_name: str = None) -> Pipeline:
    """Train/tune all configured models and return the best fitted pipeline."""
    best_pipeline, _ = train_and_select_best_model(X_train, y_train)
    return best_pipeline
