"""Load the saved model and predict heart disease for one or more patients.

Usage (from the repo root):
    python -m src.predict data/sample_request.json

The JSON file holds one patient (an object) or several (a list of objects),
each with the 13 feature columns used in training.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURES

MODEL_FILE = Path("models/model.joblib")
LABELS = {0: "no disease", 1: "disease"}


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_FILE):
    """Load the fitted pipeline once and reuse it (the API calls this a lot)."""
    return joblib.load(path)


def predict(patients: list[dict]) -> list[dict]:
    """Return prediction, label and confidence for each patient dict."""
    # Build a DataFrame with the columns in the same order as training.
    # Missing keys become NaN, and the pipeline's imputers fill them.
    df = pd.DataFrame(patients).reindex(columns=FEATURES)

    model = load_model()
    proba_disease = model.predict_proba(df)[:, 1]

    results = []
    for p in proba_disease:
        prediction = int(p >= 0.5)
        results.append(
            {
                "prediction": prediction,
                "label": LABELS[prediction],
                # Confidence = probability of the class we predicted
                "confidence": round(float(p if prediction == 1 else 1 - p), 4),
                "probability_disease": round(float(p), 4),
            }
        )
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <input.json>")
        sys.exit(1)

    data = json.loads(Path(sys.argv[1]).read_text())
    patients = data if isinstance(data, list) else [data]
    for patient, result in zip(patients, predict(patients)):
        print(json.dumps({"input": patient, "output": result}, indent=2))


if __name__ == "__main__":
    main()
