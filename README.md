# Credit Risk ML Engineering System

This repository implements a production-oriented Machine Learning system for credit risk prediction.  
The goal of this project is to demonstrate ML Engineering practices beyond model training, including reproducibility, packaging, model serving, and containerization.

---

## Overview

This project covers the full lifecycle of an ML system:

- Data loading and preprocessing pipelines
- Model training and artifact persistence
- Inference API using FastAPI
- Containerized deployment (Docker planned)
- Software engineering best practices (src layout, packaging, Git workflows)

This repository is designed as a portfolio project to transition from Data Scientist to Machine Learning Engineer.

---

## Architecture

High-level system design:

1. Training pipeline produces a serialized model artifact
2. Model artifact is loaded into a FastAPI inference service
3. The service exposes HTTP endpoints for real-time predictions
4. The service is containerized for reproducible deployment (Docker)

---

## Project Structure

credit-risk-ml-engineering/

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
    .gitignore
    README.md

---

## Setup

Create environment:

    python -m venv .venv
    .venv\Scripts\activate   (Windows)

Install package:

    pip install -e .

---

## Train Model

    python -m credit_risk_ml.train

Model artifact is stored at:

    models/credit_risk_model.joblib

---

## Run Inference API

    uvicorn credit_risk_ml.api:app --reload

Swagger documentation:

    http://127.0.0.1:8000/docs

---

## API Endpoints

GET /

Returns service health status.

POST /predict

Accepts a JSON payload with feature values and returns a credit default prediction.

---

## ML Engineering Concepts Demonstrated

- Modular Python package with src layout
- Reproducible environments using pyproject.toml
- Separation of training and inference code
- Model artifact management outside version control
- REST API for real-time inference
- Software engineering workflow using Git and virtual environments

---

## Roadmap

Planned extensions:

- Docker containerization for deployment portability
- MLflow for experiment tracking and model registry
- CI/CD pipeline with GitHub Actions
- Batch inference pipeline with orchestration (Airflow/Prefect)
- Cloud deployment (AWS/GCP/Azure)
- Model monitoring and retraining pipeline

---

## Author

Cristian Guachamin
Machine Learning Engineer in training
