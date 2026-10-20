"""Draw the project architecture diagram -> report/architecture.png

Usage (from the repo root):
    python scripts/draw_architecture.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = "report/architecture.png"

# Colours per area of the system
DEV, CI, SERVE, MON = "#dbeafe", "#fde68a", "#bbf7d0", "#fbcfe8"


def box(ax, x, y, w, h, text, color, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=color, ec="#334155", lw=1.2
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, weight="bold" if bold else "normal"
    )


def group(ax, x, y, w, h, title):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc="none", ec="#64748b", lw=1, ls="--"
        )
    )
    ax.text(x + 0.15, y + h - 0.3, title, fontsize=10, weight="bold", color="#334155", va="top")


def arrow(ax, x1, y1, x2, y2, label=None, curve=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.3,
            color="#334155",
            connectionstyle=f"arc3,rad={curve}",
        )
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, fontsize=8, ha="center", color="#475569")


fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis("off")

# 1. Development: data -> training -> MLflow
group(ax, 0.2, 4.4, 6.3, 3.9, "1. Data & experimentation (Prayogshala)")
box(ax, 0.5, 6.9, 1.8, 0.8, "UCI Heart Disease\n(id 45)", DEV)
box(ax, 2.6, 6.9, 1.6, 0.8, "download_data.py", DEV)
box(ax, 4.5, 6.9, 1.7, 0.8, "preprocess.py\nheart_clean.csv", DEV)
box(ax, 0.5, 5.2, 2.2, 1.0, "01_eda.ipynb\nEDA plots", DEV)
box(
    ax,
    3.0,
    5.0,
    3.2,
    1.3,
    "train.py\nPipeline(ColumnTransformer + model)\nGridSearchCV, stratified 5-fold\nLogReg vs RandomForest",
    DEV,
    True,
)
box(ax, 3.0, 4.6, 1.5, 0.3, "MLflow ./mlruns", "#e9d5ff")
box(ax, 4.7, 4.6, 1.5, 0.3, "models/model.joblib", "#e9d5ff")
arrow(ax, 2.3, 7.3, 2.6, 7.3)
arrow(ax, 4.2, 7.3, 4.5, 7.3)
arrow(ax, 5.3, 6.9, 5.0, 6.3)
arrow(ax, 4.6, 6.9, 1.8, 6.2, curve=0.1)

# 2. CI/CD on GitHub
group(ax, 6.9, 4.4, 7.9, 3.9, "2. CI/CD (GitHub Actions, on every push / PR)")
box(ax, 7.2, 6.9, 1.6, 0.8, "git push\nmain", CI)
box(ax, 9.1, 6.9, 1.2, 0.8, "Lint\nruff", CI)
box(ax, 10.6, 6.9, 1.4, 0.8, "Unit tests\npytest (21)", CI)
box(ax, 12.3, 7.25, 2.2, 0.6, "Train + smoke test", CI)
box(ax, 12.3, 6.5, 2.2, 0.6, "Docker build + /predict", CI)
box(
    ax,
    9.1,
    5.0,
    5.4,
    1.0,
    (
        "Artifacts per run: test-results.xml, trained model,\n"
        "mlruns/, tuning tables, training log\n"
        "Any failing step stops the pipeline (red build)"
    ),
    "#fef3c7",
)
arrow(ax, 8.8, 7.3, 9.1, 7.3)
arrow(ax, 10.3, 7.3, 10.6, 7.3)
arrow(ax, 12.0, 7.4, 12.3, 7.55)
arrow(ax, 12.0, 7.2, 12.3, 6.8)
arrow(ax, 13.4, 6.5, 13.0, 6.0)

# 3. Serving on Minikube
group(ax, 0.2, 0.2, 9.6, 3.9, "3. Serving (Docker image on Minikube Kubernetes)")
box(ax, 0.5, 2.5, 1.8, 0.8, "Client\ncurl / Swagger", SERVE)
box(ax, 2.7, 2.3, 2.2, 1.2, "Service\nLoadBalancer\nport 80 -> 8000", SERVE, True)
box(ax, 5.4, 2.55, 4.1, 0.85, "Pod 1: FastAPI + model\n/predict  /health  /metrics", SERVE)
box(ax, 5.4, 1.55, 4.1, 0.85, "Pod 2: FastAPI + model\n/predict  /health  /metrics", SERVE)
box(
    ax,
    0.5,
    0.5,
    4.4,
    0.9,
    "Deployment: 2 replicas, readiness/liveness\nprobes on /health, CPU/memory limits",
    "#dcfce7",
)
box(
    ax,
    5.4,
    0.5,
    4.1,
    0.9,
    (
        "Docker image heart-disease-api:1.0 (same Dockerfile as CI)\n"
        "python:3.12-slim + models/model.joblib, non-root"
    ),
    "#dcfce7",
)
arrow(ax, 2.3, 2.9, 2.7, 2.9, "JSON")
arrow(ax, 4.9, 3.0, 5.4, 3.0)
arrow(ax, 4.9, 2.6, 5.4, 2.0)

# 4. Monitoring
group(ax, 10.2, 0.2, 4.6, 3.9, "4. Monitoring & logging")
box(ax, 10.5, 2.4, 1.9, 0.9, "Prometheus\nscrapes every 10s", MON)
box(ax, 12.7, 2.4, 1.8, 0.9, "Grafana\ndashboard", MON)
box(
    ax,
    10.5,
    0.6,
    4.0,
    1.4,
    "JSON request logs on stdout\n(method, path, status, latency,\nprediction, confidence)\nkubectl logs",
    MON,
)
arrow(ax, 9.5, 2.95, 10.5, 2.85, "/metrics")
arrow(ax, 9.5, 1.9, 10.5, 1.5, "logs")
arrow(ax, 12.4, 2.85, 12.7, 2.85)

ax.set_title("Heart Disease MLOps - architecture", fontsize=14, weight="bold", pad=10)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"Saved {OUT}")
