from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class LearnedFusion:
    """Validation-trained fusion; missing branches are represented by an availability bit."""
    def __init__(self): self.model = LogisticRegression(max_iter=1000); self.fitted = False
    def fit(self, dsp: list[float], deep: list[float], labels: list[int]) -> "LearnedFusion":
        if not (len(dsp)==len(deep)==len(labels)): raise ValueError("fusion arrays must have equal length")
        self.model.fit(np.column_stack([dsp, deep, np.ones(len(dsp)), np.ones(len(dsp))]), labels); self.fitted=True; return self
    def predict(self, dsp: float | None, deep: float | None) -> float:
        if not self.fitted: raise RuntimeError("fusion model is unfitted")
        if dsp is None and deep is None: raise ValueError("at least one score is required")
        row=[dsp or 0.0, deep or 0.0, float(dsp is not None), float(deep is not None)]
        return float(self.model.predict_proba([row])[0,1])
