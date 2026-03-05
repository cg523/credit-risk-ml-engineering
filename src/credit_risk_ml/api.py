from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import os

# Optional: allow override of tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---- Model URI using Registry alias ----
MODEL_URI = "models:/credit_risk_model@champion"

# Load model once at startup
model = mlflow.pyfunc.load_model(MODEL_URI)


print(f"Serving model version: {model_version.version}")

app = FastAPI(title="Credit Risk Model API")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return {"default_prediction": int(prediction)}