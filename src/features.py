"""Feature definitions and the preprocessing ColumnTransformer.

The same preprocessor is used during training and inside the saved model,
so the API applies exactly the same transformations as training did.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Real measurements on a continuous scale
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Codes that stand for categories (e.g. cp 1-4 = chest pain type).
# The numbers have no real order or distance, so we one-hot encode them.
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the heart disease features."""
    numeric_pipeline = Pipeline(
        steps=[
            # The cleaned CSV has no gaps, but an API request might. The
            # median is not pulled around by outliers (e.g. chol = 564).
            ("imputer", SimpleImputer(strategy="median")),
            # Put every numeric feature on the same scale (mean 0, std 1) so
            # chol (~250) does not dominate oldpeak (~1). This matters for
            # Logistic Regression; Random Forest does not care, but it is harmless.
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            # For categories the only sensible fill is the most common value
            # (the same choice as in preprocess.py).
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # One column per category value (cp=1, cp=2, ...).
            # handle_unknown="ignore": if a request sends a code that was never
            # seen in training, all its columns become 0 instead of crashing.
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
