# Project: Heart Disease MLOps (BITS Pilani AIMLCZG523 Assignment 1)
Dataset: UCI Heart Disease (Cleveland), binary target. Python 3.12.
Layout: scripts/download_data.py, src/ (preprocess, train, predict, api),
notebooks/, tests/, k8s/, .github/workflows/ci.yml, Dockerfile,
requirements.txt (pinned), screenshots/, report/.
Rules: all scripts run from a clean venv via requirements.txt. Log every
experiment to MLflow. Keep code simple and commented so the student can
explain it. Never commit secrets. Explain what you did after each step.
Steps: 1 data+EDA, 2 pipeline+models+CV, 3 MLflow, 4 packaging,
5 pytest+GitHub Actions (lint/test/train, must fail on errors),
6 FastAPI /predict (JSON in, prediction+confidence out) + Dockerfile,
7 Minikube deployment (deployment.yaml, service.yaml), 8 /metrics with
prometheus_client + request logging, 9 report.
