# heart-disease-mlops

Predicting heart disease from patient health data (UCI Heart Disease,
Cleveland subset), built as an end-to-end MLOps project for BITS Pilani
AIMLCZG523 Assignment 1.

## Project layout
```
scripts/download_data.py   download the raw dataset from UCI
src/preprocess.py          clean data, binary target -> data/processed/
src/features.py            preprocessing ColumnTransformer (scaling + one-hot)
src/train.py               tune + compare models, log to MLflow, save best model
src/predict.py             load the saved model and predict (CLI + used by the API)
src/api.py                 FastAPI service: /predict, /health, /metrics
tests/                     21 pytest unit tests (data, features, model, API)
.github/workflows/ci.yml   GitHub Actions: lint -> test -> train, Docker build + smoke test
Dockerfile                 API container (uses requirements-serve.txt)
k8s/                       Minikube deployment + LoadBalancer service
k8s/monitoring/            Prometheus + Grafana (dashboard preconfigured)
notebooks/01_eda.ipynb     exploratory data analysis
data/raw/, data/processed/ raw and cleaned CSVs (committed)
models/                    saved best pipeline + metadata.json
report/                    final report (report.md / report.pdf), model selection, tuning tables
screenshots/               plots and screenshots used in the report
```

## Setup
Requires Python 3.12.
```bash
git clone https://github.com/khuranaj95/heart-disease-mlops.git
cd heart-disease-mlops
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline
All commands are run from the repo root with the venv active.

```bash
# 1. Download raw data -> data/raw/heart_raw.csv
python scripts/download_data.py

# 2. Clean it -> data/processed/heart_clean.csv
python src/preprocess.py

# 3. EDA notebook (plots are saved to screenshots/eda/)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

# 4. Train, tune and compare models; logs to ./mlruns and saves models/model.joblib
python -m src.train
```

## Experiment tracking (MLflow)
Runs are stored locally in `./mlruns` (not committed). To browse them:
```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri file:./mlruns --port 5000
```
Then open http://localhost:5000 (on a remote server, forward port 5000 first).
MLflow 3 blocks the plain-folder store unless `MLFLOW_ALLOW_FILE_STORE=true` is set.

The experiment `heart-disease` has one parent run per model (best settings,
CV + test metrics, ROC curve, confusion matrix, fitted pipeline) with one
nested child run for every grid-search setting tried.

## Results
| Model | CV ROC-AUC | CV Recall | Test ROC-AUC | Test Recall |
|---|---|---|---|---|
| **Logistic Regression** (selected) | 0.902 | 0.783 | 0.966 | 0.929 |
| Random Forest | 0.902 | 0.774 | 0.948 | 0.893 |

Why Logistic Regression was chosen: [report/model_selection.md](report/model_selection.md).

## Tests and linting
```bash
ruff check . && ruff format --check .
python -m pytest -v
python -m src.predict data/sample_request.json
```

## Run the API with Docker
```bash
docker build -t heart-disease-api:1.0 .
docker run -p 8000:8000 heart-disease-api:1.0
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
# -> {"prediction":1,"label":"disease","confidence":0.9856,"probability_disease":0.9856}
```
Interactive API docs: http://localhost:8000/docs

## Deploy to Minikube
```bash
minikube start --driver=docker
minikube image load heart-disease-api:1.0
kubectl apply -f k8s/                     # deployment (2 replicas) + LoadBalancer service
minikube tunnel                           # in a second terminal: assigns the external IP
kubectl get svc heart-disease-api         # EXTERNAL-IP, port 80
minikube service heart-disease-api --url  # alternative URL without the tunnel
```

## Monitoring
```bash
kubectl apply -f k8s/monitoring/
minikube service prometheus --url         # Prometheus (targets, queries)
minikube service grafana --url            # Grafana -> "Heart Disease API" dashboard
```
The API logs one JSON line per request (`kubectl logs deployment/heart-disease-api`) and
exposes Prometheus metrics at `/metrics`.

## Report
`report/report.md` (source) and `report/report.pdf`. Rebuild the PDF with
`python scripts/build_report.py` (needs Firefox); redraw the architecture diagram with
`python scripts/draw_architecture.py`.
