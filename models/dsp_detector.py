from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dsp.features import extract_features
from .base import DetectorOutput


class DSPDetector:
    def __init__(self, classifier: str = "logistic_regression"):
        choices = {"logistic_regression": LogisticRegression(max_iter=1000), "random_forest": RandomForestClassifier(n_estimators=100, random_state=42), "gradient_boosting": GradientBoostingClassifier(random_state=42), "svm": SVC(probability=True, random_state=42)}
        if classifier not in choices: raise ValueError(f"unsupported classifier: {classifier}")
        self.feature_names: list[str] = []
        self.pipeline = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("classifier", choices[classifier])])
        self.trained = False

    def fit(self, feature_rows: list[dict[str, float]], labels: list[int]) -> "DSPDetector":
        if len(feature_rows) != len(labels) or len(set(labels)) < 2: raise ValueError("need equal rows and at least two classes")
        self.feature_names = sorted(feature_rows[0])
        self.pipeline.fit([[row.get(name, np.nan) for name in self.feature_names] for row in feature_rows], labels)
        self.trained = True
        return self

    def predict_proba(self, audio: np.ndarray, sample_rate: int) -> DetectorOutput:
        if not self.trained: raise RuntimeError("DSP detector is untrained")
        row = extract_features(audio, sample_rate)
        matrix = [[row.get(name, np.nan) for name in self.feature_names]]
        score = float(self.pipeline.predict_proba(matrix)[0, 1])
        return DetectorOutput(score, "dsp-baseline-1", False, "dsp")

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle: pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "DSPDetector":
        with Path(path).open("rb") as handle: return pickle.load(handle)
