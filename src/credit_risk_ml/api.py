from fastapi import FastAPI
from joblib import load
import pandas as pd

app = FastAPI(title="Credit Risk Model API")

# Load model at startup (global, in-memory)
model = load('models/credit_risk_model.joblib')

@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: dict):
    """
    features: {"feature1": value, "feature2": value, ...}
    """
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return {"default_prediction": int(prediction)}