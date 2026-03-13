import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

from credit_risk_ml.data import load_data
from credit_risk_ml.features import preprocess
from credit_risk_ml.model import train_model

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_default_data.csv"
MODELS_PATH = PROJECT_ROOT / "models"

def main():
    print("Starting trainig pipeline")

    mlflow.set_experiment("credit-risk-training")

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

        # Log dataset
        mlflow.log_artifact(DATA_PATH)

        # Log feature names
        feature_names = list(X.columns)

        with open("features.txt", "w") as f:
            for feature in feature_names:
                f.write(f"{feature}\n")

        mlflow.log_artifact("features.txt")

        # Log model
        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id

        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="credit_risk_model"
        )

        # Create models folder
        MODELS_PATH.mkdir(exist_ok=True)

        baseline_path = MODELS_PATH / "baseline_model"

        # Remove existing model if exists
        if baseline_path.exists():
            shutil.rmtree(baseline_path)

        # Save model locally
        mlflow.sklearn.save_model(
            sk_model=model,
            path=MODELS_PATH / "baseline_model"
        )

        print(f"Model saved to {baseline_path}")
        print(f"Accuracy logged: {accuracy}")



if __name__ == "__main__":
    main()