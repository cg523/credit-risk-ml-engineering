import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

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

    mlflow.set_experiment("credit-risk-trainig")

    with mlflow.start_run():

        df = load_data(DATA_PATH)
        df = preprocess(df)

        X = df.drop('default', axis=1)
        y = df['default']

        model = train_model(X,y)

        print("Model trained successfully!")

        # Log parameters
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # Log metrics
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Create models folder
        MODELS_PATH.mkdir(exist_ok=True)

        # Save model locally
        dump(model, MODEL_PATH)

        print(f"Model saved to {MODEL_PATH}")
        print(f"Accuracy logged: {accuracy}")



if __name__ == "__main__":
    main()