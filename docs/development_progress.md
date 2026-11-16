# Development progress

## 2026-11-16

- Inspected the existing `data` branch and retained its dataset/preprocessing tests.
- Added a tested FastAPI DSP-only foundation: bounded buffers, VAD, FFT/STFT features, DSP detector contract, score fusion, Bayesian risk, REST, and WebSocket streaming.
- Added pinned core dependencies, Docker Compose for API/Redis/MongoDB/MLflow, Make targets, and initial architecture/API/privacy documentation.
- GitHub push verification is currently blocked by unavailable network/DNS; no push is claimed.

Deferred until the next implementation slice: durable Redis/Mongo adapters, MLflow/ZenML runners, ONNX adapter, full manifest/evaluation framework, Terraform, and expanded test suites.

## 2026-11-16 continuation

- Added failure-safe optional Redis/Mongo metadata adapters with TTL/index behavior and raw-audio redaction.
- Added MLflow metadata logging, ZenML-compatible training entry points, local training/evaluation/benchmark scripts, grouped zero-day split helpers, and reproducible audio augmentations.
- Added mock VoIP notification interfaces, API-key enforcement hooks, alert acknowledgement endpoints, latency registry, local WAV ingestion, expanded Terraform baseline, and restricted-environment installation guidance.
- Untrained inference now returns no DSP score rather than a hand-designed heuristic score.
