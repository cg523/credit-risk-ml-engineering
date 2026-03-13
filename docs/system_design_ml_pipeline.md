# Credit Risk ML System — Architecture Overview

This repository implements a simplified production-style Machine Learning system for credit risk prediction. The goal of the project is to demonstrate end-to-end Machine Learning Engineering practices including model training, validation, promotion, and serving.

The system follows a modular architecture separating **training**, **model registry**, **CI/CD automation**, and **serving**.

---

# System Components

The system is composed of four main layers:

1. Training Pipeline
2. Model Registry
3. CI/CD Pipeline
4. Model Serving API

---

# Training Pipeline

The training pipeline is implemented in:

```
src/credit_risk_ml/train.py
```

Responsibilities:

* Load raw dataset
* Perform preprocessing
* Train a scikit-learn model
* Log parameters and metrics to MLflow
* Register the trained model in the MLflow Model Registry

Each training execution creates:

* a new MLflow run
* a new model version in the registry

Experiment tracking is handled by MLflow.

---

# Model Registry

The system uses MLflow Model Registry to manage model versions.

Registered model name:

```
credit_risk_model
```

Model versions are created automatically after training.

Model promotion is controlled using an alias:

```
champion
```

The `champion` alias represents the model currently considered production-ready.

---

# Model Validation

Model validation is implemented in:

```
src/credit_risk_ml/validate_model.py
```

The validation step compares:

```
latest trained model
vs
current champion model
```

The comparison is based on evaluation metrics (currently accuracy).

Validation outcomes:

* If the new model improves the metric → validation passes
* If performance is worse → pipeline fails

Cold-start handling is implemented to support the case where no champion model exists.

---

# Model Promotion

Model promotion is handled by:

```
src/credit_risk_ml/set_alias.py
```

If validation succeeds, the newest model version is assigned the alias:

```
champion
```

This mechanism decouples the serving layer from training runs.

---

# Model Serving

Model serving is implemented using FastAPI.

```
src/credit_risk_ml/serving/api.py
```

Endpoints:

```
GET /
health check

POST /predict
accepts feature inputs as JSON and returns the predicted default risk.
```

The API loads the model using MLflow:

```
models:/credit_risk_model@champion
```

This ensures the API always serves the currently promoted production model.

---

# CI/CD Pipeline

The project uses GitHub Actions for CI/CD.

Pipeline defined in:

```
.github/workflows/ci.yml
```

Pipeline stages:

```
push
 ↓
tests
 ↓
train
 ↓
validate
 ↓
promotion
```

Steps executed:

1. Run unit tests (pytest)
2. Train a new model
3. Validate the model against the current champion
4. Promote the model if validation passes

---

# Model Loading Strategy

Model loading depends on the execution environment.

Implemented in:

```
src/credit_risk_ml/serving/model_loader.py
```

Behavior:

CI environment
→ loads a baseline model stored in the repository

Local / production environment
→ loads the champion model from MLflow registry

This allows CI tests to run without depending on the MLflow registry state.

---

# Key Architectural Decisions

Training and serving are fully separated.

Model serving is controlled through the MLflow registry using aliases rather than run IDs.

CI pipelines simulate model promotion logic using ephemeral MLflow environments.

A baseline model is versioned in the repository to allow reproducible CI tests.

MLflow tracking is currently local (no remote tracking server).

---

# Future Extensions

Possible improvements to the system include:

* Remote MLflow tracking server
* Dataset versioning
* Feature store integration
* Monitoring and drift detection
* Scheduled retraining pipelines
