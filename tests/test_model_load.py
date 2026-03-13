from credit_risk_ml.serving.model_loader import load_model


def test_model_load():

    model = load_model()

    assert model is not None