import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd


def compare_models():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    runs = client.search_runs(experiment.experiment_id)

    data = []

    for run in runs:
        data.append({
            "run_name": run.data.tags.get("mlflow.runName"),
            "rmse": run.data.metrics.get("rmse"),
            "mae": run.data.metrics.get("mae")
        })

    df = pd.DataFrame(data)
    print("\nModel Comparison:\n")
    print(df.sort_values("rmse"))


if __name__ == "__main__":
    compare_models()
