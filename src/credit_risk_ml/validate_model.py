import sys
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


MODEL_NAME = "credit_risk_model"
METRIC = "accuracy"

client = MlflowClient()

def main():

    try:
        champion = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="champion"
        )

    except MlflowException:
        print("No champion model found. Skipping validation.")
        return

    # Get champion
    champion = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias="champion"
    )

    champion_run = client.get_run(champion.run_id)
    champion_metric = champion_run.data.metrics[METRIC]

    # Get latest model
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    latest_version = max(
        versions,
        key=lambda v: int(v.version)
    )

    latest_run = client.get_run(latest_version.run_id)
    latest_metric = latest_run.data.metrics[METRIC]

    print(f"Champion accuracy: {champion_metric}")
    print(f"Latest accuracy: {latest_metric}")

    if latest_metric > champion_metric:
        print("New model is better. Promotion allowed.")
        sys.exit(0)

    print("New model is not better. Promotion rejected.")
    sys.exit(1)


if __name__ == "__main__":
    main()