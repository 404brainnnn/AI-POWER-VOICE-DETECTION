from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ScoreCalibrator:
    """Validation-only probability calibration; never fit this on test data."""
    def __init__(self, method: str = "platt"):
        if method not in {"platt", "isotonic"}: raise ValueError("method must be platt or isotonic")
        self.method, self.model, self.fitted = method, None, False

    def fit(self, scores: list[float], labels: list[int]) -> "ScoreCalibrator":
        if len(scores) != len(labels) or len(set(labels)) < 2: raise ValueError("calibration needs two classes")
        x, y = np.asarray(scores).reshape(-1, 1), np.asarray(labels)
        self.model = LogisticRegression(C=1e6).fit(x, y) if self.method == "platt" else IsotonicRegression(out_of_bounds="clip").fit(x.ravel(), y)
        self.fitted = True
        return self

    def transform(self, scores: list[float] | np.ndarray) -> np.ndarray:
        if not self.fitted: raise RuntimeError("calibrator is unfitted")
        x = np.asarray(scores)
        return self.model.predict_proba(x.reshape(-1, 1))[:, 1] if self.method == "platt" else self.model.predict(x)
