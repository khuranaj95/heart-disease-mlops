"""Train, tune and compare models, log everything to MLflow, save the best one.

Usage (from the repo root):
    python -m src.train
"""

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw plots to files only (no window needed, works in CI)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, build_preprocessor

DATA_FILE = Path("data/processed/heart_clean.csv")
MODEL_DIR = Path("models")
PLOT_DIR = Path("screenshots/models")
TUNING_DIR = Path("report/tuning")
EXPERIMENT_NAME = "heart-disease"
TRACKING_URI = "file:./mlruns"  # local folder store, no server needed
RANDOM_STATE = 42  # fixed seed so every run gives the same split and results

# The metrics we report, both in cross-validation and on the test set
SCORING = ["accuracy", "precision", "recall", "roc_auc"]

# Each candidate model and a small grid of settings to try.
# "model__C" means: parameter C of the pipeline step called "model".
CANDIDATES = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "param_grid": {
            # C = inverse regularisation strength. Small C = simpler model.
            "model__C": [0.01, 0.1, 1.0, 10.0],
        },
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators": [100, 300],  # number of trees
            "model__max_depth": [None, 5, 10],  # limit tree depth to reduce overfitting
            "model__min_samples_leaf": [1, 5],  # minimum patients per leaf
        },
    },
}


def evaluate_on_test(pipeline, X_test, y_test) -> dict:
    """Score a fitted pipeline on the held-out test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # probability of disease
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def save_plots(name, pipeline, X_test, y_test) -> list[Path]:
    """Save ROC curve and confusion matrix PNGs for the test set."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    roc_path = PLOT_DIR / f"{name}_roc_curve.png"
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_title(f"ROC curve - {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)

    cm_path = PLOT_DIR / f"{name}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, display_labels=["No disease", "Disease"], ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    return [roc_path, cm_path]


def train_one(name, spec, X_train, y_train, X_test, y_test) -> dict:
    """Tune one model with GridSearchCV, evaluate it, and log it as an MLflow run."""
    pipeline = Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", spec["model"])])

    # Stratified 5-fold: every fold keeps the same disease/no-disease ratio
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        spec["param_grid"],
        cv=cv,
        scoring=SCORING,
        refit="roc_auc",  # choose the best settings by ROC-AUC
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    # Mean cross-validation score of the best settings, for every metric
    i = search.best_index_
    cv_scores = {m: search.cv_results_[f"mean_test_{m}"][i] for m in SCORING}
    test_scores = evaluate_on_test(best, X_test, y_test)

    # Full grid results: one row per setting tried, sorted best first.
    # Saved to report/ so the tuning process is documented in the repo.
    # Values are written as text so max_depth=None shows as "None", not blank.
    grid = pd.DataFrame(
        [{k.replace("model__", ""): str(v) for k, v in p.items()} for p in search.cv_results_["params"]]
    )
    for m in SCORING:
        grid[f"cv_{m}_mean"] = search.cv_results_[f"mean_test_{m}"]
        grid[f"cv_{m}_std"] = search.cv_results_[f"std_test_{m}"]
    grid["rank"] = search.cv_results_["rank_test_roc_auc"]
    grid = grid.sort_values("rank")
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = TUNING_DIR / f"{name}_grid_results.csv"
    grid.round(4).to_csv(grid_path, index=False)

    with mlflow.start_run(run_name=name) as run:
        # One nested "child" run per grid setting, so every setting we tried
        # is visible in MLflow, not just the winner.
        for i, params in enumerate(search.cv_results_["params"]):
            short = {k.replace("model__", ""): v for k, v in params.items()}
            with mlflow.start_run(run_name=f"{name}_grid_{i:02d}", nested=True):
                mlflow.log_param("model_type", name)
                mlflow.log_params(short)
                mlflow.log_metrics({f"cv_{m}": search.cv_results_[f"mean_test_{m}"][i] for m in SCORING})
                mlflow.log_metric("rank", search.cv_results_["rank_test_roc_auc"][i])

        # The parent run holds the best setting and its final evaluation
        mlflow.log_param("model_type", name)
        mlflow.log_param("grid_size", len(search.cv_results_["params"]))
        mlflow.log_artifact(str(grid_path), artifact_path="tuning")
        mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_param("cv_folds", 5)
        mlflow.log_metrics({f"cv_{m}": v for m, v in cv_scores.items()})
        mlflow.log_metrics({f"test_{m}": v for m, v in test_scores.items()})

        for plot in save_plots(name, best, X_test, y_test):
            mlflow.log_artifact(str(plot), artifact_path="plots")

        # Log the whole fitted pipeline (preprocessing + model) as an MLflow model.
        # The signature records the expected input columns and output type.
        # Columns are declared as floats so inputs with missing values are allowed.
        X_float = X_train.astype(float)
        signature = infer_signature(X_float, best.predict(X_train))
        mlflow.sklearn.log_model(
            best,
            name="model",
            signature=signature,
            input_example=X_float.head(3),
            # MLflow saves sklearn models with skops, which only loads types we
            # trust. numpy.dtype comes from the OneHotEncoder settings, and Tree
            # is how Random Forest stores its trees. We trust both because we
            # create these files ourselves in this script.
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id

    return {
        "name": name,
        "pipeline": best,
        "best_params": search.best_params_,
        "cv": cv_scores,
        "test": test_scores,
        "run_id": run_id,
    }


def main():
    df = pd.read_csv(DATA_FILE)
    X, y = df[FEATURES], df[TARGET]

    # Hold out 20% for a final, untouched test. stratify=y keeps the class
    # ratio the same in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # MLflow 3 prefers a database store and blocks the plain-folder store
    # unless we opt in. A folder is simpler to inspect, so we opt in.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    for name, spec in CANDIDATES.items():
        print(f"\nTuning {name} ...")
        result = train_one(name, spec, X_train, y_train, X_test, y_test)
        print(f"  best params: {result['best_params']}")
        results.append(result)

    # Print a comparison table
    rows = []
    for r in results:
        for split in ["cv", "test"]:
            rows.append({"model": r["name"], "split": split, **r[split]})
    table = pd.DataFrame(rows).set_index(["model", "split"]).round(3)
    print("\n" + table.to_string())
    table.to_csv(TUNING_DIR / "model_comparison.csv")

    # Pick the winner by cross-validated ROC-AUC (not test score: the test set
    # must stay unseen while choosing, otherwise it is no longer a fair test).
    best = max(results, key=lambda r: r["cv"]["roc_auc"])
    print(
        f"\nBest model: {best['name']} "
        f"(CV ROC-AUC {best['cv']['roc_auc']:.3f}, CV recall {best['cv']['recall']:.3f})"
    )

    # Save the winning pipeline and a metadata file that describes it
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_DIR / "model.joblib")

    metadata = {
        "model_name": best["name"],
        "best_params": best["best_params"],
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "cv_metrics": best["cv"],
        "test_metrics": best["test"],
        "mlflow_run_id": best["run_id"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "mlflow": mlflow.__version__,
            "joblib": joblib.__version__,
        },
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=float)
    print(f"Saved {MODEL_DIR / 'model.joblib'} and {MODEL_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
