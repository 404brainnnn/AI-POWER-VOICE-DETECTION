# Model integration

`models.base.DeepLearningDetector` defines the future adapter contract. `models.onnx_adapter.ONNXDetector` loads a supplied ONNX artifact, validates its output, and requires explicit calibration metadata. No backbone, weights, mock research score, or automatic model registration is included.

## Stage 1 neural models

`models.neural.BaselineCNN` supports `lcnn` and `resnet18` backbones. It computes an on-the-fly Hann-windowed STFT with `n_fft=512`, `win_length=400`, and `hop_length=160`, applies log magnitude and per-utterance normalization, and returns `(logits[B,2], embedding[B,D])`.

`models.neural.XLSRAASIST` lazily loads `facebook/wav2vec2-xls-r-300m`, learns a weighted sum across transformer hidden states, and applies a graph-attention temporal head. It supports feature-extractor freezing, layer truncation, and separate encoder/head learning rates. The Stage 1 tests construct a tiny local Wav2Vec2 configuration and do not download the 300M checkpoint.

Model selection is a single YAML switch in `configs/model.yaml`; use `models.factory.build_detector(config)` after loading the YAML. Labels are `real=0` and `ai=1`.
