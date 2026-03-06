import os
import joblib
import mlflow.pyfunc


def test_model_load():

    if os.getenv("CI"):
        model = joblib.load("models/credit_risk_model.joblib")
    else:
        model = mlflow.pyfunc.load_model("models:/credit_risk_model@champion")

    assert model is not None