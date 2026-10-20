"""Tests for src/features.py (the preprocessing ColumnTransformer)."""

import numpy as np
import pandas as pd
import pytest

from src.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, build_preprocessor

DATA_FILE = "data/processed/heart_clean.csv"


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(DATA_FILE)


def test_feature_lists_cover_all_columns(data):
    assert set(FEATURES) == set(data.columns) - {"target"}
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)


def test_output_shape(data):
    pre = build_preprocessor()
    out = pre.fit_transform(data[FEATURES])
    # 5 scaled numeric columns + one column per category value
    n_categories = sum(data[c].nunique() for c in CATEGORICAL_FEATURES)
    assert out.shape == (len(data), len(NUMERIC_FEATURES) + n_categories)


def test_numeric_features_are_scaled(data):
    pre = build_preprocessor().fit(data[FEATURES])
    scaled = pre.named_transformers_["num"].transform(data[NUMERIC_FEATURES])
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-6)


def test_handles_missing_and_unknown_values(data):
    pre = build_preprocessor().fit(data[FEATURES])
    row = data[FEATURES].iloc[[0]].astype(float)
    row.loc[:, "chol"] = np.nan  # missing numeric -> median
    row.loc[:, "thal"] = np.nan  # missing category -> most frequent
    row.loc[:, "cp"] = 99  # never-seen category -> all zeros, no crash
    out = pre.transform(row)
    assert not np.isnan(np.asarray(out.todense() if hasattr(out, "todense") else out)).any()
