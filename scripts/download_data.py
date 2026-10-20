"""Download the UCI Heart Disease dataset and save it as a raw CSV.

Usage (from the repo root):
    python scripts/download_data.py
"""

from pathlib import Path

from ucimlrepo import fetch_ucirepo

# The UCI repository id for "Heart Disease" (Cleveland subset)
UCI_DATASET_ID = 45
RAW_DIR = Path("data/raw")
RAW_FILE = RAW_DIR / "heart_raw.csv"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading UCI dataset id={UCI_DATASET_ID} ...")
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)

    # ucimlrepo returns the features (X) and target (y) separately;
    # join them so the raw file contains everything in one table.
    features = dataset.data.features
    target = dataset.data.targets
    df = features.join(target)

    # Save exactly what we downloaded, with no changes (cleaning happens later)
    df.to_csv(RAW_FILE, index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} columns to {RAW_FILE}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
