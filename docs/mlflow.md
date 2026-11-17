# MLflow

The Compose stack exposes MLflow at `http://localhost:5000`. Install `pip install -e '.[mlops]'` and set `MLFLOW_TRACKING_URI`. Audio is never logged automatically; log only configurations, metrics, plots, and approved model artifacts. Untrained/mock models must not be registered as production versions.

For a metadata-only smoke log: `python scripts/mlflow_log.py --roc-auc 0.5`. The value is an example input, not a measured research result.
