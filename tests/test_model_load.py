import mlflow.pyfunc


MODEL_URI = "models:/credit_risk_model@champion"


def test_model_load():
    model = mlflow.pyfunc.load_model(MODEL_URI)
    assert model is not None