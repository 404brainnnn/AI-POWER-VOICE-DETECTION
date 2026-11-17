from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectorOutput:
    score: float
    model_version: str
    calibrated: bool
    mode: str


class DeepLearningDetector(ABC):
    @abstractmethod
    def load_model(self) -> None: ...
    @abstractmethod
    def predict_proba(self, audio: np.ndarray, sample_rate: int) -> DetectorOutput: ...
    @abstractmethod
    def get_model_metadata(self) -> dict[str, str]: ...
