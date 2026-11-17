# Stage 1: neural model setup

Install the ML dependencies on Python 3.11 with a PyTorch wheel appropriate for the target CUDA runtime:

```bash
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-ml.txt
python -m pytest tests/test_neural_models.py tests/test_xlsr_head.py
```

The baseline tests use random tensors, verify both classifier output and embedding shapes, and run a tiny-batch overfit check. The XLS-R test uses a tiny randomly initialized Wav2Vec2 encoder; it is not a research result and does not validate XLS-R accuracy.

To use the real encoder, select `xlsr_aasist` in `configs/model.yaml`. The first construction downloads the Hugging Face checkpoint unless it is already in the configured Hugging Face cache. In production, pin the model revision and mirror/cache the artifact in S3 or an approved model registry.
