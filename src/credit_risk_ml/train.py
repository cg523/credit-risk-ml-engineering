
from credit_risk_ml.data import load_data
from credit_risk_ml.features import preprocess
from credit_risk_ml.model import train_model

from joblib import dump
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_default_data.csv"
MODELS_PATH = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_PATH / "credit_risk_model.joblib"

def main():
    print("Starting trainig pipeline")

    df = load_data(DATA_PATH)
    df = preprocess(df)

    X = df.drop('default', axis=1)
    y = df['default']

    model = train_model(X,y)
    print("Model trained successfully!")

    # Create models folder
    MODELS_PATH.mkdir(exist_ok=True)

    # Save model
    dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")



if __name__ == "__main__":
    main()