from mlflow.tracking import MlflowClient

client = MlflowClient()

client.set_registered_model_alias(
    name="credit_risk_model",
    alias="champion",
    version="4"
)

print("Alias 'champion' assigned to version 4")