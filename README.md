# AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

This repository contains a production-oriented DSP-only research foundation:

- FastAPI REST and WebSocket APIs under `/api/v1`.
- Validated 8/16/48 kHz audio, bounded RAM-only buffers, 3-second sliding windows, and energy VAD.
- Deterministic FFT/STFT, spectral, Mel/DCT, pitch-proxy, phase, and speech-activity features.
- Trainable scikit-learn DSP detector, calibration utility, weighted fusion, and discounted Bayesian risk.
- Grouped manifest/split and evaluation metric utilities.
- Optional ONNX and ZenML boundaries without a deep model or synthetic research results.
- Docker Compose services for API, Redis, MongoDB, and MLflow.

The deep-learning backbone, remote SSH dataset access, AWS provisioning, and frontend are intentionally deferred. No raw audio is persisted. See the `docs/` directory.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-core.txt
uvicorn app.main:app --reload
```

Then open `http://localhost:8000/docs`. For local infrastructure use `docker compose up --build`.
