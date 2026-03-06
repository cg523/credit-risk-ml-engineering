import os
import mlflow.pyfunc


def test_model_load():

    if os.getenv("CI"):
        model_uri = "models/credit_risk_model.joblib"
    else:
        model_uri = "models:/credit_risk_model@champion"

    model = mlflow.pyfunc.load_model(model_uri)

    assert model is not None