"""Tests for src/preprocess.py (data cleaning)."""

import numpy as np
import pandas as pd

from src.preprocess import clean


def make_raw():
    """A tiny raw dataset with the same columns as the UCI file."""
    return pd.DataFrame(
        {
            "age": [63, 67, 41, 50],
            "sex": [1, 1, 0, 0],
            "cp": [1, 4, 2, 3],
            "trestbps": [145, 160, 130, 120],
            "chol": [233, 286, 204, 219],
            "fbs": [1, 0, 0, 0],
            "restecg": [2, 2, 2, 0],
            "thalach": [150, 108, 172, 158],
            "exang": [0, 1, 0, 0],
            "oldpeak": [2.3, 1.5, 1.4, 1.6],
            "slope": [3, 2, 1, 2],
            # '?' is how the original UCI file marks missing values
            "ca": [0, "?", 0, np.nan],
            "thal": [6, 3, "?", 3],
            "num": [0, 2, 0, 4],
        }
    )


def test_no_missing_values_after_cleaning():
    df = clean(make_raw())
    assert df.isna().sum().sum() == 0


def test_missing_values_filled_with_mode():
    df = clean(make_raw())
    # ca: known values are 0, 0 -> mode 0; thal: 6, 3, 3 -> mode 3
    assert df.loc[1, "ca"] == 0 and df.loc[3, "ca"] == 0
    assert df.loc[2, "thal"] == 3


def test_target_is_binary():
    df = clean(make_raw())
    # num 0 -> 0, num 1-4 -> 1
    assert df["target"].tolist() == [0, 1, 0, 1]
    assert "num" not in df.columns


def test_ca_and_thal_are_integers():
    df = clean(make_raw())
    assert df["ca"].dtype.kind == "i"
    assert df["thal"].dtype.kind == "i"


def test_original_dataframe_not_modified():
    raw = make_raw()
    clean(raw)
    assert "num" in raw.columns and (raw["ca"] == "?").any()
