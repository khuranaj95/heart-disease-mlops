# Container for the heart disease prediction API.
# Build:  docker build -t heart-disease-api:1.0 .
# Run:    docker run -p 8000:8000 heart-disease-api:1.0

# Small official Python image, same version used for training
FROM python:3.12-slim

# Don't write .pyc files; print logs straight away (no buffering)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first: this layer is cached and only rebuilt
# when requirements-serve.txt changes, not on every code change.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Application code and the trained model
COPY src/ src/
COPY models/ models/

# Run as a normal user, not root (basic container security)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Docker marks the container unhealthy if /health stops answering
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# --no-access-log: our middleware already writes one JSON log line per request
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
