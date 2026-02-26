from fastapi import FastAPI
from joblib import load
import pandas as pd
from pathlib import Path
import os

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "credit_risk_model.joblib"))

# Load model at startup
model = load(MODEL_PATH)

app = FastAPI(title="Credit Risk Model API")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return {"default_prediction": int(prediction)}
