"""FastAPI service that serves the heart disease model.

Endpoints:
    GET  /health   -> is the service up and the model loaded?
    POST /predict  -> JSON patient data in, prediction + confidence out
    GET  /metrics  -> Prometheus metrics (request counts, latency, predictions)

Run locally (from the repo root):
    uvicorn src.api:app --port 8000
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.predict import MODEL_FILE, load_model, predict

# ---------- Logging: one JSON line per request, written to stdout ----------
# stdout is where Docker and Kubernetes collect logs (docker logs / kubectl logs).
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("heart-api")

# ---------- Prometheus metrics ----------
REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_duration_seconds", "Request latency in seconds", ["endpoint"])
PREDICTIONS = Counter("model_predictions_total", "Predictions made, by predicted label", ["label"])
CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Confidence of each prediction",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)


# ---------- Request / response schemas ----------
class Patient(BaseModel):
    """The 13 clinical features. Ranges reject obviously wrong input with a 422 error."""

    age: int = Field(ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(description="Chest pain type (1 typical .. 4 asymptomatic)")
    trestbps: float = Field(ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(description="Resting ECG result")
    thalach: float = Field(ge=50, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(description="Exercise induced angina")
    oldpeak: float = Field(ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(description="Slope of the peak exercise ST segment")
    ca: Literal[0, 1, 2, 3] = Field(description="Number of major vessels coloured (0-3)")
    thal: Literal[3, 6, 7] = Field(description="3 normal, 6 fixed defect, 7 reversible defect")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
                "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
            }
        }
    }  # fmt: skip


class Prediction(BaseModel):
    prediction: int = Field(description="1 = heart disease, 0 = no heart disease")
    label: str
    confidence: float = Field(description="Probability of the predicted class (0.5-1.0)")
    probability_disease: float


# ---------- App ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, so the first request is not slow
    # and a missing model file stops the container immediately.
    load_model()
    logger.info(json.dumps({"event": "startup", "model": str(MODEL_FILE)}))
    yield


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    """Time every request, count it for Prometheus and write a log line."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    endpoint = request.url.path
    REQUESTS.labels(request.method, endpoint, response.status_code).inc()
    LATENCY.labels(endpoint).observe(duration)
    logger.info(
        json.dumps(
            {
                "event": "request",
                "method": request.method,
                "path": endpoint,
                "status": response.status_code,
                "duration_ms": round(duration * 1000, 2),
                "client": request.client.host if request.client else None,
            }
        )
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": load_model() is not None}


@app.post("/predict", response_model=Prediction)
def predict_endpoint(patient: Patient):
    result = predict([patient.model_dump()])[0]

    PREDICTIONS.labels(result["label"]).inc()
    CONFIDENCE.observe(result["confidence"])
    # Log only the model output, not the patient's health data (privacy)
    logger.info(json.dumps({"event": "prediction", **result}))
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
