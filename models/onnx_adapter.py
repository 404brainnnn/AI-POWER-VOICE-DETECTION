from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import DeepLearningDetector, DetectorOutput


class ONNXDetector(DeepLearningDetector):
    """Adapter for a future supplied model. It refuses uncalibrated production use."""
    def __init__(self, model_path: str | Path, calibrated: bool = False, providers: list[str] | None = None):
        self.model_path, self.calibrated, self.providers = Path(model_path), calibrated, providers
        self.session = None

    def load_model(self) -> None:
        if not self.model_path.is_file(): raise FileNotFoundError(self.model_path)
        try: import onnxruntime as ort
        except ImportError as exc: raise RuntimeError("install the optional onnx extra") from exc
        self.session = ort.InferenceSession(str(self.model_path), providers=self.providers or ["CPUExecutionProvider"])

    def predict_proba(self, audio: np.ndarray, sample_rate: int) -> DetectorOutput:
        if self.session is None: raise RuntimeError("ONNX model is not loaded")
        if not self.calibrated: raise RuntimeError("explicit calibration metadata is required")
        inputs = self.session.get_inputs()
        output = self.session.run(None, {inputs[0].name: np.asarray(audio, dtype=np.float32)[None, :]})[0]
        score = float(np.asarray(output).reshape(-1)[-1])
        if not 0 <= score <= 1: raise ValueError("ONNX output must be a probability in [0, 1]")
        return DetectorOutput(score, self.model_path.stem, True, "onnx")

    def get_model_metadata(self) -> dict[str, str]:
        return {"path": str(self.model_path), "calibrated": str(self.calibrated), "status": "loaded" if self.session else "unloaded"}
