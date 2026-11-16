# Model integration

`models.base.DeepLearningDetector` defines the future adapter contract. `models.onnx_adapter.ONNXDetector` loads a supplied ONNX artifact, validates its output, and requires explicit calibration metadata. No backbone, weights, mock research score, or automatic model registration is included.
