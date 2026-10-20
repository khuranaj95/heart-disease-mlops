"""Tests for src/predict.py (loading the saved model and predicting)."""

import json

import pytest

from src.predict import MODEL_FILE, predict

pytestmark = pytest.mark.skipif(not MODEL_FILE.exists(), reason="run python -m src.train first")


@pytest.fixture(scope="module")
def samples():
    with open("data/sample_request.json") as f:
        return json.load(f)


def test_output_format(samples):
    for result in predict(samples):
        assert set(result) == {"prediction", "label", "confidence", "probability_disease"}
        assert result["prediction"] in (0, 1)
        assert 0.5 <= result["confidence"] <= 1.0
        assert 0.0 <= result["probability_disease"] <= 1.0


def test_label_matches_probability(samples):
    for r in predict(samples):
        assert r["prediction"] == int(r["probability_disease"] >= 0.5)
        assert r["label"] == ("disease" if r["prediction"] == 1 else "no disease")


def test_known_patients(samples):
    # sample 0 is a healthy patient, sample 1 has heart disease (from the dataset)
    results = predict(samples)
    assert results[0]["prediction"] == 0
    assert results[1]["prediction"] == 1


def test_missing_feature_still_predicts(samples):
    patient = dict(samples[0])
    del patient["chol"]
    assert predict([patient])[0]["prediction"] in (0, 1)
