from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import os

# ---- Required environment variable ----
RUN_ID = os.getenv("MODEL_RUN_ID")

if not RUN_ID:
    raise ValueError(
        "MODEL_RUN_ID environment variable not set. "
        "Set it to a valid MLflow run ID."
    )

MODEL_URI = f"runs:/{RUN_ID}/model"

# Load model from MLflow
model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(title="Credit Risk Model API")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return {"default_prediction": int(prediction)}
