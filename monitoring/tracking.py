from __future__ import annotations

from typing import Any


def log_run(params: dict[str, Any], metrics: dict[str, float], artifacts: list[str] | None = None) -> str | None:
    """Log metadata only; audio is deliberately excluded from artifacts."""
    try:
        import mlflow
    except ImportError:
        return None
    with mlflow.start_run() as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        for artifact in artifacts or []: mlflow.log_artifact(artifact)
        return run.info.run_id
