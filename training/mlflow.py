from __future__ import annotations

import json
from pathlib import Path


def mlflow_run(config, train_callable):
    """Run a callable inside MLflow when installed; otherwise fail explicitly."""
    try:
        import mlflow
    except ImportError as exc: raise RuntimeError("install the optional MLOps dependencies") from exc
    if config.mlflow_tracking_uri: mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({"model": config.model, "seed": config.seed, "loss": config.loss, "amp": config.amp})
        result=train_callable(run)
        if isinstance(result,dict): mlflow.log_metrics({key:float(value) for key,value in result.items() if isinstance(value,(float,int))})
        return run.info.run_id


class ActiveRunLogger:
    def log_metrics(self, metrics: dict, step: int) -> None:
        import mlflow
        mlflow.log_metrics({key: float(value) for key,value in metrics.items()}, step=step)


def register_checkpoint(run_id: str, checkpoint_path: str, model_name: str, thresholds_path: str | None = None) -> str:
    import mlflow
    with mlflow.start_run(run_id=run_id):
        artifact_dir=Path(checkpoint_path).parent
        mlflow.log_artifact(checkpoint_path, artifact_path="model")
        if thresholds_path: mlflow.log_artifact(thresholds_path, artifact_path="model")
        model_uri=f"runs:/{run_id}/model"
        result=mlflow.register_model(model_uri,model_name)
        return result.version
