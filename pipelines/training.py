from __future__ import annotations

from dsp.features import extract_features
from models.dsp_detector import DSPDetector


def train_dsp_feature_rows(feature_rows: list[dict[str, float]], labels: list[int], classifier: str = "logistic_regression") -> DSPDetector:
    """Reusable ZenML-compatible training function; caller controls grouped splits."""
    return DSPDetector(classifier).fit(feature_rows, labels)
