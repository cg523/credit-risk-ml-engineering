import pandas as pd
from credit_risk_ml.features import preprocess

def test_preprocess():
    df = pd.DataFrame({"a": [1, None]})
    result = preprocess(df)
    assert result.isnull().sum().sum() == 0