```md
# Credit Risk ML Engineering System

This repository implements a production-oriented Machine Learning system for credit risk prediction.  
The goal of this project is to demonstrate ML Engineering practices beyond model training, including reproducibility, packaging, model serving, and containerization.

---

## Overview

This project covers the full lifecycle of an ML system:

- Data loading and preprocessing pipelines
- Model training and artifact persistence
- Inference API using FastAPI
- Containerized deployment using Docker
- Software engineering best practices (src layout, packaging, Git workflows)

This repository is designed as a portfolio project to transition from Data Scientist to Machine Learning Engineer.

---

## Architecture

High-level system design:

1. Training pipeline produces a serialized model artifact
2. Model artifact is stored outside the Docker image
3. Model artifact is loaded into a FastAPI inference service at runtime
4. The service exposes HTTP endpoints for real-time predictions
5. The service is containerized for reproducible deployment

**Key ML Engineering Principle:** Code, dependencies, and model artifacts are decoupled.

---

## Project Structure

```

credit-risk-ml-engineering/

```
src/
    credit_risk_ml/
        data.py        # data loading
        features.py    # preprocessing pipeline
        model.py        # model definition
        train.py         # training entrypoint
        api.py           # inference API

data/                   # ignored (raw datasets)
models/                 # ignored (trained artifacts)
notebooks/
tests/

pyproject.toml
requirements.txt
Dockerfile
.gitignore
README.md
```

````

---

## Setup (Local Development)

Create environment:

```bash
python -m venv .venv
.venv\\Scripts\\activate   # Windows
````

Install package:

```bash
pip install -e .
```

---

## Train Model

```bash
python -m credit_risk_ml.train
```

Model artifact is stored at:

```bash
models/credit_risk_model.joblib
```

---

## Run Inference API (Local)

```bash
uvicorn credit_risk_ml.api:app --reload
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

## Docker (Reproducible Inference Service)

### Build Docker Image

```bash
docker build -t credit-risk-api .
```

### Run Container with External Model Artifact

**Linux / macOS:**

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_PATH=/app/models/credit_risk_model.joblib \
  credit-risk-api
```

**Windows PowerShell:**

```powershell
docker run -p 8000:8000 `
  -v ${PWD}/models:/app/models `
  -e MODEL_PATH=/app/models/credit_risk_model.joblib `
  credit-risk-api
```

### Key Concept

The Docker image does **not** contain the model.
The model is mounted at runtime as an external artifact, enabling model updates without rebuilding the image.

---

## API Endpoints

### GET /

Returns service health status.

### POST /predict

Accepts a JSON payload with feature values and returns a credit default prediction.

---

## ML Engineering Concepts Demonstrated

* Modular Python package with src layout
* Reproducible environments using pyproject.toml
* Separation of training and inference code
* Model artifact management outside version control
* Runtime model loading via Docker volumes
* REST API for real-time inference
* Containerized reproducible deployments
* Software engineering workflow using Git and virtual environments

---

## Roadmap

Planned extensions:

* MLflow for experiment tracking and model registry
* CI/CD pipeline with GitHub Actions
* Batch inference pipeline with orchestration (Airflow/Prefect)
* Cloud deployment (AWS/GCP/Azure)
* Model monitoring and retraining pipeline

---

## Author

Cristian Guachamin
Machine Learning Engineer in training

```
```
