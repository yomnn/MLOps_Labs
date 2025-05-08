import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("https://dagshub.com/yomna.wael.mu/MLOps_Labs.mlflow")
client = MlflowClient()

# Create a new experiment
experiment_id = client.create_experiment("titanic-mlflow-experiment")
print(f"Experiment created with ID: {experiment_id}")
