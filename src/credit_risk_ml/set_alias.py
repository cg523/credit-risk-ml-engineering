from mlflow.tracking import MlflowClient

MODEL_NAME = "credit_risk_model"


def main():

    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    latest_version = max(
        versions,
        key=lambda v: int(v.version)
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=latest_version.version
    )

    print(f"Champion alias set to version {latest_version.version}")


if __name__ == "__main__":
    main()