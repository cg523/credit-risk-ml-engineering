import os
import mlflow.pyfunc


def get_model_uri():
    """
    Resolve the model URI depending on the environment.
    """

    if os.getenv("CI"):
        return "models/baseline_model"

    return "models:/credit_risk_model@champion"


def load_model():
    """
    Load the ML model using MLflow.
    """

    model_uri = get_model_uri()

    model = mlflow.pyfunc.load_model(model_uri)

    print(f"Model loaded from: {model_uri}")

    return model