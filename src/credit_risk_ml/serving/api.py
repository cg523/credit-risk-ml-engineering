from fastapi import FastAPI
import pandas as pd

from credit_risk_ml.serving.model_loader import load_model

app = FastAPI(title="Credit Risk Model API")

model = load_model()


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: dict):

    df = pd.DataFrame([features])

    prediction = model.predict(df)[0]

    return {"default_prediction": int(prediction)}