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

## 2026-11-17

- Added trainable CSV-backed PyTorch datasets (`location,label`, human=0, AI=1), CNN/XLS-R training loop, AMP, weighted CE/OC-Softmax, checkpoint resume, EER early stopping, calibration, thresholds, evaluation, and MLflow/ZenML entry points.
- Added model checkpoint loading, bounded asynchronous inference batching, `/score`, `/healthz`, `/readyz`, `/metrics`, PCM16 WebSocket streaming, TorchScript/ONNX export and latency benchmark scripts.
- Added research-analysis notebook for class balance, waveform, log-Mel, and class-mean spectrogram figures.
- Added GPU ECS EC2 Terraform baseline, private subnets/NAT, optional ALB TLS, S3 model artifact access, deployment rollback circuit breaker, ECR workflow, and Locust load test.
- Source syntax and notebook JSON validation passed. ML runtime tests and Terraform validation remain unexecuted because PyTorch/Transformers/pytest/Terraform are unavailable in the current environment.
- Git stage commit was blocked by a read-only `.git` index (`index.lock` cannot be created).
