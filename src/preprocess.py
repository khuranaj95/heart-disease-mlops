"""Clean the raw heart disease data and save a model-ready CSV.

Usage (from the repo root):
    python src/preprocess.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILE = Path("data/raw/heart_raw.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_FILE = PROCESSED_DIR / "heart_clean.csv"

# Columns that contain missing values in the Cleveland data
COLUMNS_WITH_MISSING = ["ca", "thal"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataframe."""
    df = df.copy()

    # The original UCI file marks missing values with '?'. ucimlrepo usually
    # converts them to NaN already, but we replace '?' anyway so this works on
    # either version of the file.
    df = df.replace("?", np.nan)
    for col in COLUMNS_WITH_MISSING:
        df[col] = pd.to_numeric(df[col])

    # Missing-value choice: fill 'ca' and 'thal' with their most common value
    # (the mode) instead of dropping rows.
    # - Only 6 of 303 values are missing, so filling barely changes the data.
    # - Both columns are categories (ca = number of vessels 0-3,
    #   thal = 3/6/7), so the mode is a valid category. A mean like 0.67
    #   vessels would not make sense.
    # - The dataset is small, so we keep all 303 patients.
    for col in COLUMNS_WITH_MISSING:
        mode_value = df[col].mode()[0]
        n_missing = df[col].isna().sum()
        df[col] = df[col].fillna(mode_value)
        print(f"Filled {n_missing} missing '{col}' values with mode = {mode_value}")

    # These are whole-number categories, so store them as integers
    df[COLUMNS_WITH_MISSING] = df[COLUMNS_WITH_MISSING].astype(int)

    # Target: the original 'num' column is 0 (no disease) to 4 (severe).
    # The assignment is a yes/no problem, so 0 -> 0 and 1-4 -> 1.
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    return df


def main():
    df_raw = pd.read_csv(RAW_FILE)
    print(f"Loaded {df_raw.shape[0]} rows from {RAW_FILE}")

    df_clean = clean(df_raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_FILE, index=False)

    print(f"Remaining missing values: {df_clean.isna().sum().sum()}")
    print("Target counts (0 = no disease, 1 = disease):")
    print(df_clean["target"].value_counts().sort_index().to_string())
    print(f"Saved {df_clean.shape[0]} rows x {df_clean.shape[1]} columns to {PROCESSED_FILE}")


if __name__ == "__main__":
    main()
