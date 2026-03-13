def test_import_training_pipeline():
    from credit_risk_ml.model import train_model
    assert train_model is not None