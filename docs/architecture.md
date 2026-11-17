# Architecture

The current implementation is a privacy-preserving, DSP-only FastAPI service. Audio is validated, retained in a bounded in-memory buffer, gated by energy VAD, and evaluated in 3-second sliding windows. Named Fourier/spectral features feed a diagnostic DSP path and a sequential Bayesian risk engine. No raw waveform is written to Redis, MongoDB, MLflow, or logs.

The `models.DeepLearningDetector` contract and future ONNX adapter boundary are intentionally present but no backbone or synthetic model is shipped. Window evidence is discounted because overlapping windows are dependent; the discount must be selected on validation data before research use.
