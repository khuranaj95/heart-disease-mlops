"""Tests for src/train.py: pipeline building, metrics and a quality gate."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.features import FEATURES, TARGET, build_preprocessor
from src.train import CANDIDATES, evaluate_on_test

DATA_FILE = "data/processed/heart_clean.csv"


def load():
    df = pd.read_csv(DATA_FILE)
    return df[FEATURES], df[TARGET]


def test_both_models_are_configured():
    assert {"logistic_regression", "random_forest"} <= set(CANDIDATES)
    for spec in CANDIDATES.values():
        # every grid key must point at the "model" step of the pipeline
        assert all(k.startswith("model__") for k in spec["param_grid"])


def test_metrics_are_between_0_and_1():
    X, y = load()
    pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    scores = evaluate_on_test(pipe, X, y)
    assert set(scores) == {"accuracy", "precision", "recall", "roc_auc"}
    assert all(0 <= v <= 1 for v in scores.values())


def test_model_quality_gate():
    """Fail the build if a simple model can no longer reach a reasonable ROC-AUC."""
    X, y = load()
    pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    assert auc >= 0.85, f"CV ROC-AUC dropped to {auc:.3f}"
