"""Tests for src/api.py using FastAPI's TestClient (no server needed)."""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.predict import MODEL_FILE

pytestmark = pytest.mark.skipif(not MODEL_FILE.exists(), reason="run python -m src.train first")

VALID_PATIENT = {
    "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0, "restecg": 2,
    "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3,
}  # fmt: skip


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs the startup code (model loading)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=VALID_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 1 and body["label"] == "disease"
    assert 0.5 <= body["confidence"] <= 1.0


def test_invalid_value_is_rejected(client):
    r = client.post("/predict", json={**VALID_PATIENT, "age": -5})
    assert r.status_code == 422


def test_missing_field_is_rejected(client):
    patient = {k: v for k, v in VALID_PATIENT.items() if k != "thal"}
    assert client.post("/predict", json=patient).status_code == 422


def test_metrics_count_predictions(client):
    client.post("/predict", json=VALID_PATIENT)
    text = client.get("/metrics").text
    assert 'model_predictions_total{label="disease"}' in text
    assert "api_requests_total" in text
